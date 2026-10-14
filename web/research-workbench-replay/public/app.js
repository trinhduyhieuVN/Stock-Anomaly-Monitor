"use strict";
const UI=window.StockMonitorUI;
const $ = id => document.getElementById(id);
const state = {catalog:null,tape:null,data:null,frame:0,selected:0,playing:false,timer:null,
  generation:0,abort:null,modelAbort:null,modelCache:new Map(),backend:null,modelKey:null,
  modelError:null,modelPending:false,modelTimer:null,mode:"observations",paint:0,animation:0,
  chartType:"line",viewLimit:0,hover:null,peerTapes:new Map(),peerCache:new Map(),peerErrors:[]};
try{const saved=localStorage.getItem('stock-monitor.chart-type');if(['line','candles'].includes(saved))state.chartType=saved;}catch{}
const number = (v,d=2) => v==null||!Number.isFinite(Number(v)) ? "Chưa có" : Number(v).toLocaleString("vi-VN",{minimumFractionDigits:d,maximumFractionDigits:d});
const time = stamp => new Date(stamp).toLocaleTimeString("vi-VN",{timeZone:"Asia/Ho_Chi_Minh",hour:"2-digit",minute:"2-digit"});
const knownTime = stamp => new Date(stamp).toLocaleTimeString("vi-VN",{timeZone:"Asia/Ho_Chi_Minh",hour:"2-digit",minute:"2-digit",second:"2-digit"});
const names = {FPT:"Công nghệ",HPG:"Thép",VCB:"Ngân hàng"};
const companyNames={FPT:'Tập đoàn FPT',HPG:'Tập đoàn Hòa Phát',VCB:'Vietcombank'};
const companyLogos={FPT:'/logos/FPT.svg',HPG:'/logos/HPG.png',VCB:'/logos/VCB.png'};
function logo(symbol){if(!companyLogos[symbol])return document.createDocumentFragment();const el=document.createElement('img');el.className='company-logo';el.width=64;el.height=36;el.alt='Logo '+companyNames[symbol];el.src=companyLogos[symbol];el.addEventListener('error',()=>{el.hidden=true;});return el;}
const kinds = UI.decisions;
function paragraphs(target,lines){target.replaceChildren();lines.filter(Boolean).forEach(line=>{const p=document.createElement("p");p.textContent=line;target.append(p);});}
function options(target,rows,value){target.replaceChildren();rows.forEach(([name,v])=>{const o=document.createElement("option");o.value=v;o.textContent=name;target.append(o);});if(rows.some(r=>r[1]===value))target.value=value;}
function query(scope){const p=new URLSearchParams();Object.entries(scope).forEach(([k,v])=>{if(v!=null)p.set(k,String(v));});return p.toString();}
function tapeScope(){return {source:$("source").value,symbol:state.symbol||"FPT",session:$("session").value,model:$("model").value};}
function scope(){const f=state.tape.frames[state.frame];return {...tapeScope(),replay:state.frame<state.tape.frames.length-1,step:f.step};}
function scopeKey(){return query(scope());}
function stop(sync=true){clearTimeout(state.timer);state.timer=null;state.playing=false;$("play").textContent="Phát Replay";if(sync)maybeSyncModel();}
async function request(path,opts={}){const response=await fetch(path,opts);const data=await response.json();if(!response.ok)throw Error(data.error||"Máy chủ local chưa sẵn sàng.");return data;}
function sourceControls(){const source=state.catalog.sources[$("source").value];options($("session"),source.sessions.map(v=>[source.session_labels?.[v]||UI.date(v),v]),source.default_session||($("source").value==="synthetic"?"2025-02-03":source.sessions.at(-1)));options($("model"),Object.entries(source.models).map(([label,v])=>[UI.modelLabel(v,label),v]));state.symbol=source.symbols[0]||"FPT";}
function invalidateModel(){clearTimeout(state.modelTimer);state.modelTimer=null;state.modelAbort?.abort();state.modelAbort=null;state.modelKey=null;state.backend=null;state.modelError=null;state.modelPending=false;}
async function load(){
  stop(false);invalidateModel();state.modelCache.clear();state.tape=null;state.data=null;state.hover=null;state.peerTapes.clear();state.peerErrors=[];$("chart-tooltip").hidden=true;
  const generation=++state.generation;state.abort?.abort();state.abort=new AbortController();
  $("watchlist").hidden=true;$("content").hidden=true;$("error").hidden=true;$("loading").hidden=false;$("workbench").setAttribute("aria-busy","true");
  $("source-status").textContent="Đang kiểm tra nguồn và phiên đang chọn…";
  try{const tape=await request('/api/replay-tape?'+query(tapeScope()),{signal:AbortSignal.any([state.abort.signal,AbortSignal.timeout(60000)])});
    if(generation!==state.generation)return;
    if(tape.live!==false||tape.model_scores_included!==false||!tape.frames.length)throw Error("Tape offline không hợp lệ");
    await loadPeerTapes(tape,generation);if(generation!==state.generation)return;
    state.tape=tape;state.frame=tape.frames.length-1;state.selected=tape.frames.at(-1).bars.length-1;
    $("content").hidden=false;applyFrame(false);renderSymbols();$("watchlist").hidden=false;maybeSyncModel();
  }catch(e){if(generation!==state.generation||e.name==="AbortError")return;state.tape=null;state.data=null;$("watchlist").hidden=true;$("content").hidden=true;$("error-copy").textContent="Không đọc được dữ liệu: "+e.message;$("error").hidden=false;$("source-status").textContent="Dữ liệu chưa sẵn sàng. Không giữ biểu đồ nguồn trước.";}
  finally{if(generation===state.generation){$("loading").hidden=true;$("workbench").setAttribute("aria-busy","false");}}
}
// Select a verified receipt frame, not the final (possibly revised) view filtered into the past.
function frameAt(tape,cutoff){if(!tape)return null;const t=Date.parse(cutoff);let lo=0,hi=tape.frames.length-1,best=-1;
  while(lo<=hi){const mid=Math.floor((lo+hi)/2);if(Date.parse(tape.frames[mid].as_of)<=t){best=mid;lo=mid+1;}else hi=mid-1;}
  return best<0?null:tape.frames[best];}
function sameSegment(a,b){return a&&b&&a.segment_id===b.segment_id&&Date.parse(b.timestamp)>Date.parse(a.timestamp)&&Date.parse(b.timestamp)-Date.parse(a.timestamp)<=300000;}
// Shading follows only observed adjacent bars, never bridges a gap or changes prices.
function areaPath(rows,x,y,base){let path='',start=0;for(let i=0;i<rows.length;i++){
  if(i===rows.length-1||!sameSegment(rows[i],rows[i+1])){if(i>start)path+='M'+x(start)+' '+base+' L'+rows.slice(start,i+1).map((b,j)=>x(start+j)+' '+y(b)).join(' L')+' L'+x(i)+' '+base+' Z ';start=i+1;}}
  return path.trim();}
function chartWindow(total,selected,limit){if(![0,12,24].includes(limit))throw Error('Phạm vi biểu đồ không hợp lệ');
  if(!limit||total<=limit)return {start:0,end:total};const last=total-limit;
  const start=selected>=last?last:Math.max(0,Math.min(last,selected-Math.floor(limit/2)));return {start,end:start+limit};}
// Display salience only: never alter descriptive rules, event rows or AI evidence.
function importantObservationIndices(items,start,end){const candidates=items.slice(start,end).map(o=>({index:o.index,
  strength:Math.max(o.priceEvent&&Number.isFinite(o.change)?Math.abs(o.change)/.6:0,o.volumeEvent&&Number.isFinite(o.ratio)?o.ratio/2:0)}))
  .filter(o=>o.strength>=1).sort((a,b)=>b.strength-a.strength||a.index-b.index),chosen=[];
  for(const o of candidates){if(chosen.every(index=>Math.abs(index-o.index)>=3))chosen.push(o.index);if(chosen.length===3)break;}
  return new Set(chosen);}
async function loadPeerTapes(main,generation){const entries=await Promise.all(state.catalog.sources[main.source].symbols.map(async symbol=>{
  const registered=state.catalog.sources[main.source].peer_manifest_by_symbol;
  const key=main.source+'/'+main.session+'/'+symbol;try{let tape=symbol===main.symbol?main:state.peerCache.get(key);
    const expected=registered?.[symbol]||main.provenance.manifest_sha256;
    if(registered&&registered[main.symbol]!==main.provenance.manifest_sha256)throw Error('Bảng giá không khớp bundle đã đăng ký');
    if(!tape||tape.provenance.manifest_sha256!==expected)tape=await request('/api/replay-tape?'+query({source:main.source,symbol,session:main.session,model:state.catalog.sources[main.source].default_model||(main.source==='historical'?'robust_zscore':'v2')}),{signal:state.abort.signal});
    if(tape.symbol!==symbol||tape.session!==main.session||tape.source!==main.source||tape.live!==false||tape.model_scores_included!==false||!tape.frames.length||tape.provenance.manifest_sha256!==expected)throw Error('Phạm vi giá không khớp');
    if(generation===state.generation){state.peerCache.set(key,tape);if(state.peerCache.size>12)state.peerCache.delete(state.peerCache.keys().next().value);}return[symbol,tape];
  }catch(e){return[symbol,null];}}));if(generation!==state.generation)return;state.peerTapes=new Map(entries);state.peerErrors=entries.filter(([,t])=>!t).map(([s])=>s);}
// Same descriptive rules as the first prototype. No model event IDs or scores.
function observations(rows){return rows.map((b,i)=>{const prior=rows.slice(Math.max(0,i-12),i).map(r=>r.volume).sort((a,b)=>a-b),n=prior.length;
  const median=n?(prior[Math.floor((n-1)/2)]+prior[Math.floor(n/2)])/2:null;
  const change=i&&rows[i-1].close>0?(b.close/rows[i-1].close-1)*100:null,ratio=median>0?b.volume/median:null;
  const priceEvent=change!==null&&Math.abs(change)>=.6,volumeEvent=ratio!==null&&ratio>=2;
  return {index:i,change,ratio,reference:n,priceEvent,volumeEvent,
    label:priceEvent?(volumeEvent?"Giá và khối lượng":"Biến động giá"):"Khối lượng tăng",
    value:priceEvent?`${change>=0?'+':''}${number(change)}%`:number(ratio,1)+"×"};});}
function modelWanted(){return state.data?.model_scoring_available!==false;}
function maybeSyncModel(){if(state.data&&modelWanted()&&!state.playing){clearTimeout(state.modelTimer);state.modelTimer=setTimeout(()=>{state.modelTimer=null;if(state.data&&modelWanted()&&!state.playing)syncModel();},150);}}
async function syncModel(){
  if(!state.data||state.playing||!modelWanted())return;
  const key=scopeKey(),generation=state.generation;if(state.modelKey===key&&(state.modelPending||state.backend))return;
  state.modelAbort?.abort();state.modelAbort=new AbortController();state.modelKey=key;state.modelPending=true;state.modelError=null;state.backend=null;
  renderModel();if(state.mode==="models")renderEvents();
  try{let data=state.modelCache.get(key);if(!data){data=await request('/api/snapshot?'+key,{signal:AbortSignal.any([state.modelAbort.signal,AbortSignal.timeout(60000)])});
      state.modelCache.set(key,data);if(state.modelCache.size>12)state.modelCache.delete(state.modelCache.keys().next().value);}
    if(generation!==state.generation||!state.data||key!==scopeKey()||state.playing)return;
    if(data.as_of!==state.data.as_of||data.source!==state.data.source||data.symbol!==state.data.symbol||data.model!==state.data.model||data.session!==state.data.session)throw Error("Phạm vi model không khớp nến đang xem");
    state.backend=data;state.modelPending=false;renderDetails();renderEvents();renderChart(false);
  }catch(e){if(generation!==state.generation||!state.data||key!==scopeKey()||e.name==="AbortError")return;state.modelPending=false;state.modelError=e.message;renderModel();if(state.mode==="models")renderEvents();}
}
function backendEvent(){const b=state.data?.bars[state.selected];return b&&state.backend?.events.find(e=>Date.parse(e.decision_bar_end||e.event_end)===Date.parse(b.timestamp));}
function renderModel(){
  const data=state.data;if(!data)return;const event=backendEvent(),detail=event?state.backend.details[event.event_id]:null;
  $("model").disabled=!modelWanted();document.querySelector('.exports').hidden=!modelWanted();
  if(!modelWanted()){$("model-result").textContent='Chưa có mô hình cho '+data.symbol;$("model-result").dataset.kind='unavailable';$("model-status").textContent='Chỉ xem dữ liệu giá; chưa chấm bất thường.';$("score").textContent='Chưa huấn luyện';$("threshold").textContent='Chưa có';$("model-overview").hidden=state.mode!=='models';$("model-overview").textContent='Mã này chưa có mô hình đã kiểm chứng.';$("retry-model").hidden=true;$("model-retry-inline").hidden=true;$("forecast-info").hidden=true;$("version-info").hidden=true;$("model-explanation").textContent='Không áp dụng mô hình hoặc ngưỡng của FPT, HPG, VCB cho mã mới.';paragraphs($("method-copy"),[]);paragraphs($("data-copy"),['Dữ liệu lịch sử, chưa có thời điểm nhận từng nến; quyền nguồn và đơn vị vẫn chưa xác nhận.','Hash dữ liệu: '+data.provenance.manifest_sha256]);return;}
  const status=state.playing?"Tạm dừng để đọc kết quả.":state.modelError?"Chưa đọc được mô hình: "+state.modelError:state.modelPending||state.modelTimer?"Đang đọc kết quả…":state.backend?"Đến "+time(state.backend.as_of):"Chờ đồng bộ kết quả.";
  $("model-status").textContent=status;
  $("model-result").textContent=state.playing?'Chờ tạm dừng':state.modelError?'Chưa đọc được kết quả':event?kinds[event.kind]:'Chờ đồng bộ';$("model-result").dataset.kind=event?.kind||'pending';
  $("model-explanation").textContent=state.playing?'Tạm dừng để đọc điểm tại mốc đang xem.':!event?'Chưa có điểm cho nến này.':event.kind==='normal'?'Điểm không vượt ngưỡng của mô hình đang chọn.':event.kind==='insufficient_data'?'Chưa đủ lịch sử hợp lệ để chấm nến này.':'Mô hình đánh dấu để xem lại, không phải tín hiệu mua hoặc bán.';
  $("model-overview").hidden=state.mode!=="models";
  const events=state.backend?.events||[],scored=new Set(events.filter(e=>e.score!=null).map(e=>Date.parse(e.decision_bar_end||e.event_end))).size;
  const unavailable=new Set(events.filter(e=>e.kind==='insufficient_data').map(e=>Date.parse(e.decision_bar_end||e.event_end))).size;
  const flagged=events.filter(e=>['point','collective'].includes(e.kind)).length;
  const episodeCopy=state.backend?.model_metadata?` · ${state.backend.episodes.length} nhóm sự kiện`:"";
  $("model-overview").textContent=state.backend?`${scored}/${data.bars.length} nến có điểm · ${flagged} kết quả cần xem${episodeCopy}`+(unavailable?` · ${unavailable} nến chưa đủ dữ liệu`:""):status;
  $("retry-model").hidden=!state.modelError;
  $("model-retry-inline").hidden=!state.modelError||state.mode!=='models';
  const summaries={robust_zscore:'So đặc trưng nến với trung vị và độ phân tán đã học.',isolation_forest:'Tìm tổ hợp đặc trưng dễ tách khỏi số đông.',lstm:'Tái tạo chuỗi 6 nến; điểm là sai số ở nến cuối.',v2:'So điểm và đoạn với mẫu giả lập, không dùng để chứng minh hiệu quả thị trường.',ai_context:'GRU12 đọc lịch sử và bối cảnh; bộ chọn phân biệt nến đơn lẻ với đoạn liên tiếp.',watchlist_gru6:'GRU6 được huấn luyện riêng cho VNM, ACB, TCB, MBB và SSI. Đang xem tập hiệu chỉnh, chưa kiểm định độc lập.'};
  paragraphs($("method-copy"),[summaries[data.model]||'Cấu hình đối chứng của GRU12.',...(detail?.method||[]),"Điểm mô hình không phải xác suất giá tăng. Điểm thiếu không được thay bằng 0."]);
  $("score").textContent=event?.score==null?(event?.kind==="insufficient_data"?"Chưa đủ dữ liệu":state.backend?"Chưa có điểm":"Chưa có kết quả"):number(event.score,5);
  $("threshold").textContent=event?.threshold==null?"Không áp dụng / chưa có":number(event.threshold,5);
  paragraphs($("data-copy"),[data.source_mode==='HISTORICAL_REPLAY'?'Quyền nguồn và đơn vị chưa xác nhận; chỉ dùng trên máy này. Thời gian phát lại giả định theo nến, không có thời điểm nhận từng nến.':'Thời gian giả lập, không phải độ trễ nguồn thực.',`Hash dữ liệu: ${data.provenance.manifest_sha256}`]);
  $("forecast-info").hidden=data.source!=="synthetic"&&!state.backend?.model_metadata;
  $("forecast-basis").textContent=data.source==='synthetic'?'Mốc lợi suất 0 · dữ liệu giả lập.':state.backend?.model_metadata?.forecast_algorithm==='zero_return'?'Giữ mốc lợi suất 0 vì Ridge chưa vượt tiêu chí tập kiểm định.':'Ridge · dự báo sau 1 hoặc 3 nến trên dữ liệu lịch sử.';
  paragraphs($("forecasts"),(detail?.forecasts||[]).map(f=>`${f.horizon_bars} nến · đích ${time(f.target_bar_end)} · dự báo ${number(100*f.prediction,4)}% · ${f.status==='mature'?'Đã có kết quả':f.status==='pending'?'Đang chờ kết quả':f.status==='censored_boundary'?'Qua ranh giới phiên; không đối chiếu':'Thiếu nến; chưa đối chiếu'}${f.realized_return==null?'':'; thực hiện '+number(100*f.realized_return,4)+'%; sai lệch '+number(100*f.residual,4)+'%'}.`));
  if(!detail?.forecasts?.length)paragraphs($("forecasts"),["Chưa có mốc đối chiếu đã đồng bộ cho nến này."]);
  $("version-info").hidden=!detail?.initial_event;
  paragraphs($("event-versions"),(detail?.versions||[]).map(v=>`v${v.version} · ${time(v.available_at)} · ${v.stage==='initial'?'Ban đầu: '+kinds[v.kind]:'Đã đủ '+v.post_evidence.horizon_bars+' nến sau quyết định; giữ nguyên đánh giá ban đầu.'}`));
}
function renderDetails(){const data=state.data,b=data.bars[state.selected],o=data.observations[state.selected];if(!b)return;
  $("selection-time").textContent=`${data.symbol} · ${time(b.timestamp)}`;
  $("selection-kind").textContent=state.mode==='models'?(backendEvent()?kinds[backendEvent().kind]:"Chưa có tín hiệu đồng bộ"):(o.priceEvent||o.volumeEvent?o.label:"Chưa vượt quy tắc quan sát");
  for(const k of ['open','high','low','close'])$(k).textContent=number(b[k]);$("volume").textContent=number(b.volume,0);
  $("return").textContent=o.change==null?"Chưa có nến trước":`${o.change>=0?'+':''}${number(o.change)}%`;
  $("volume-ratio").textContent=o.ratio==null?"Chưa có nền hợp lệ":number(o.ratio,1)+"×";
  paragraphs($("description"),[o.change==null?"Nến đầu tiên trong phần đang xem.":o.change===0?"Giá đóng không đổi so với nến có dữ liệu trước.":`Giá đóng ${o.change>0?'tăng':'giảm'} ${number(Math.abs(o.change))}% so với nến có dữ liệu trước.`,o.ratio==null?"Chưa có nền khối lượng hợp lệ để so sánh.":`Khối lượng bằng ${number(o.ratio,1)}× trung vị ${o.reference} nến trước có dữ liệu.`]);
  $("read-selection").textContent="Đọc nến "+time(b.timestamp)+" đang chọn";
  $("select-prev").disabled=state.selected===0;$("select-next").disabled=state.selected===data.bars.length-1;renderModel();
}
function select(index){if(!state.data)return;state.selected=Math.max(0,Math.min(index,state.data.bars.length-1));state.hover=null;renderChart(false);renderDetails();renderEvents();}
function renderEvents(){const data=state.data,target=$("events"),filter=$("event-filter").value;target.replaceChildren();
  const entries=state.mode==='observations'?data.observations.filter(o=>(o.priceEvent||o.volumeEvent)&&(filter==='all'||filter==='price'&&o.priceEvent||filter==='volume'&&o.volumeEvent)).map(o=>({index:o.index,label:o.label,value:o.value,id:String(o.index)})):
    (state.backend?.events||[]).slice().reverse().filter(e=>filter==='all'||filter==='flagged'&&['point','collective'].includes(e.kind)||e.kind===filter).map(e=>({index:data.bars.findIndex(b=>Date.parse(b.timestamp)===Date.parse(e.decision_bar_end||e.event_end)),label:kinds[e.kind],value:e.score!=null?`Điểm ${number(e.score,5)}${e.version>1?' · v'+e.version:''}`:e.kind==='insufficient_data'?'Chưa đủ dữ liệu':e.stage==='initial'?'Ban đầu':`Có đối chiếu · v${e.version}`,id:e.event_id}));
  entries.filter(e=>e.index>=0).forEach(e=>{const b=document.createElement('button');b.className='event';b.dataset.eventId=e.id;b.setAttribute('aria-pressed',String(e.index===state.selected));
    for(const[tag,txt]of[['time',time(data.bars[e.index].timestamp)],['strong',e.label],['span',e.value]]){const el=document.createElement(tag);el.textContent=txt;if(tag==='span'&&state.mode==='observations')el.className='observation-value';b.append(el);}
    b.addEventListener('click',()=>{select(e.index);target.querySelector('[data-event-id="'+e.id+'"]')?.focus();});target.append(b);});
  $("no-events").hidden=target.childElementCount>0;$("no-events").textContent=state.mode==='models'?(!modelWanted()?'Chưa chấm mô hình cho mã này.':state.modelError?'Chọn Thử lại mô hình để đọc lại kết quả.':!state.backend?state.playing?'Tạm dừng để đọc tín hiệu tại mốc đang xem.':'Đang đọc kết quả…':filter==='flagged'?'Không có kết quả vượt ngưỡng trong phần đã chấm. Chọn Tất cả nến để xem điểm và phần thiếu dữ liệu.':'Không có kết quả theo bộ lọc này.'):'Không có điểm theo bộ lọc này.';
}
function setMode(mode){state.mode=mode;$("observations-tab").setAttribute('aria-pressed',String(mode==='observations'));$("models-tab").setAttribute('aria-pressed',String(mode==='models'));$("rule-copy").hidden=mode!=='observations';$("events-title").textContent=mode==='observations'?'Biến động quan sát':'Tín hiệu mô hình';$("mark-label").textContent=mode==='observations'?'Điểm nổi bật':$("events-title").textContent;
  for(const id of ['nav-overview','nav-signals'])$(id).removeAttribute('aria-current');$(mode==='models'?'nav-signals':'nav-overview').setAttribute('aria-current','location');
  options($("event-filter"),mode==='observations'?[['Tất cả loại','all'],['Biến động giá','price'],['Khối lượng tăng','volume']]:[['Tất cả nến','all'],['Được đánh dấu','flagged'],['Điểm đơn lẻ','point'],['Đoạn liên tiếp','collective'],['Không vượt ngưỡng','normal'],['Chưa đủ dữ liệu','insufficient_data']]);
  if(state.data){renderEvents();renderDetails();renderChart(false);maybeSyncModel();}}
function renderSymbols(){const data=state.data,target=$("symbols"),search=$("search").value.trim().toUpperCase();target.replaceChildren();const symbols=state.catalog.sources[data.source].symbols;$("symbol-count").textContent=symbols.length+" mã";
  symbols.filter(s=>s.includes(search)).forEach(s=>{const button=document.createElement('button');button.className='symbol';button.dataset.symbol=s;button.setAttribute('aria-pressed',String(s===data.symbol));button.setAttribute('aria-label','Xem mã '+s);const source=state.catalog.sources[data.source],priceOnly=source.price_only_symbols?.includes(s);button.disabled=!!priceOnly&&!source.price_only_sessions[s]?.includes(data.session);
    const top=document.createElement('span');top.className='symbol-top';const strong=document.createElement('strong');strong.textContent=s;const value=document.createElement('span');value.className='symbol-price';top.append(logo(s),strong,value);
    const bottom=document.createElement('span');bottom.className='symbol-bottom';const name=document.createElement('span');name.textContent=priceOnly?'Chỉ xem giá':names[s]||s;const delta=document.createElement('span');delta.className='symbol-delta';bottom.append(name,delta);
    const spark=svgElement('svg',{class:'quote-spark',viewBox:'0 0 150 56','aria-hidden':'true'}),defs=svgElement('defs'),gradient=svgElement('linearGradient',{id:'quote-fill-'+s,x1:0,y1:0,x2:0,y2:1});gradient.append(svgElement('stop',{offset:0,'stop-color':'currentColor','stop-opacity':.22}),svgElement('stop',{offset:1,'stop-color':'currentColor','stop-opacity':0}));defs.append(gradient);spark.append(defs,svgElement('path',{class:'spark-area',fill:'url(#quote-fill-'+s+')'}),svgElement('path',{class:'spark-line'}),svgElement('circle',{r:3}));const stamp=document.createElement('span');stamp.className='symbol-time';
    button.append(top,bottom,spark,stamp);button.addEventListener('click',async()=>{state.symbol=s;await load();if(state.data?.symbol===s)target.querySelector('[data-symbol="'+s+'"]')?.focus();});target.append(button);});$("no-symbols").hidden=target.childElementCount>0;updateSymbolQuote();}
function updateSymbolQuote(){if(!state.data)return;const cutoff=state.data.as_of;
  $("symbols").querySelectorAll('.symbol').forEach(button=>{const frame=frameAt(state.peerTapes.get(button.dataset.symbol),cutoff),rows=frame?.bars||[],p=button.querySelector('.symbol-price'),d=button.querySelector('.symbol-delta'),stamp=button.querySelector('.symbol-time'),spark=button.querySelector('.quote-spark'),path=spark.querySelector('.spark-line'),area=spark.querySelector('.spark-area'),dot=spark.querySelector('circle');
    button.dataset.asOf=frame?.as_of||'';if(!rows.length){delete button.dataset.quote;delete button.dataset.visibleBars;p.textContent='Chưa có';d.textContent='';stamp.textContent='Chưa có nến đã biết tại T';path.setAttribute('d','');area.setAttribute('d','');dot.setAttribute('visibility','hidden');return;}
    const last=rows.at(-1),change=rows[0].open?(last.close/rows[0].open-1)*100:null;p.textContent=number(last.close);d.textContent=change==null?'Chưa có':(change>=0?'+':'')+number(change)+'%';d.className='symbol-delta '+(change>=0?'positive':'negative');stamp.textContent='Đến '+time(last.timestamp);button.dataset.quote=String(last.close);button.dataset.visibleBars=String(rows.length);
    const values=rows.map(b=>b.close),low=Math.min(...values),high=Math.max(...values),range=Math.max(high-low,.001),x=i=>3+144*i/Math.max(rows.length-1,1),y=b=>46-38*(b.close-low)/range;
    path.setAttribute('d',rows.map((b,i)=>(i&&sameSegment(rows[i-1],b)?'L':'M')+x(i)+' '+y(b)).join(' '));area.setAttribute('d',areaPath(rows,x,y,54));spark.style.color=change>=0?'var(--up)':'var(--down)';dot.setAttribute('cx',String(x(rows.length-1)));dot.setAttribute('cy',String(y(last)));dot.setAttribute('visibility',rows.length===1?'visible':'hidden');
  });$("quote-status").textContent=state.peerErrors.length?'Chưa đọc được '+state.peerErrors.join(', ')+'. Không dùng giá của phiên trước.':'';}
function applyFrame(animate=true){const tape=state.tape,f=tape.frames[state.frame];state.data={...tape,...f,replay:state.frame<tape.frames.length-1};state.data.observations=observations(f.bars);state.selected=Math.min(state.selected,Math.max(0,f.bars.length-1));
  const last=f.bars.at(-1),first=f.bars[0],change=first.open?(last.close/first.open-1)*100:null;
  $("chart-heading").textContent=tape.symbol;$("instrument-name").textContent=UI.date(tape.session)+" · Nến 5 phút";$("last-price").textContent=number(last.close);$("day-change").textContent=change==null?'':(change>=0?'+':'')+number(change)+'%';$("day-change").className=change>=0?'positive':'negative';
  const company=$("company-logo");company.hidden=!companyLogos[tape.symbol];if(companyLogos[tape.symbol]&&company.getAttribute('src')!==companyLogos[tape.symbol]){company.src=companyLogos[tape.symbol];company.alt='Logo '+companyNames[tape.symbol];}
  $("known-time").textContent='Đã biết đến '+knownTime(f.as_of);$("unit").textContent=tape.source==='synthetic'?'Đơn vị giả lập':'Đơn vị nguồn';
  $("source-status").textContent=UI.sourceStatus(tape.source_mode);
  if(tape.evaluation_split==='validation_calibration')$("source-status").textContent+=' · Tập hiệu chỉnh GRU6, không phải kiểm định độc lập';
  const extra=state.catalog.sources[tape.source].extra_session_error;if(extra)$("source-status").textContent+=' · Phiên dao động chưa sẵn sàng: '+extra;
  $("session-date").textContent=UI.date(tape.session);$("session-clock").textContent='Nến 5 phút';state.hover=null;
  $("timeline").max=tape.frames.length;$("timeline").value=state.frame+1;$("timeline").setAttribute('aria-valuetext',f.bars.length+' nến, đã biết đến '+time(f.as_of));$("bar-count").textContent=f.bars.length+' / '+tape.frames.at(-1).bars.length+' nến';
  $("back").disabled=state.frame===0;$("next").disabled=state.frame===tape.frames.length-1;$("full").disabled=state.frame===tape.frames.length-1;
  for(const kind of ['bars','events'])$("export-"+kind).href='/api/export?'+scopeKey()+'&kind='+kind;
  updateSymbolQuote();renderChart(animate);renderDetails();renderEvents();
}
function setFrame(index,{playing=false}={}){if(!state.tape)return;if(!playing)stop(false);invalidateModel();state.frame=Math.max(0,Math.min(Number(index),state.tape.frames.length-1));state.selected=state.tape.frames[state.frame].bars.length-1;
  cancelAnimationFrame(state.paint);state.paint=requestAnimationFrame(()=>{applyFrame(true);if(!playing)maybeSyncModel();});}
function play(){if(!state.tape)return;if(state.playing){stop();renderModel();return;}state.playing=true;invalidateModel();$("play").textContent='Tạm dừng';if(state.frame===state.tape.frames.length-1)setFrame(0,{playing:true});else renderModel();
  const tick=()=>{if(!state.playing)return;if(state.frame>=state.tape.frames.length-1){stop();return;}setFrame(state.frame+1,{playing:true});if(state.frame>=state.tape.frames.length-1){state.playing=false;$("play").textContent='Phát Replay';requestAnimationFrame(maybeSyncModel);}else state.timer=setTimeout(tick,650);};state.timer=setTimeout(tick,650);
}
// Persistent SVG and keyed candle nodes. Geometry updates are coalesced in rAF;
// no chart teardown, fetch, busy overlay or cold-load animation during Replay.
const svgElement=(name,attrs={},text)=>{const el=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text!==undefined)el.textContent=text;return el;};
const chartView={svg:null,candles:new Map(),ticks:new Map(),grids:[],labels:[],positions:[]};
function setChartType(type){if(!['line','candles'].includes(type))throw Error('Loại biểu đồ không hợp lệ');state.chartType=type;
  try{localStorage.setItem('stock-monitor.chart-type',type);}catch{}if(state.data)renderChart(false);}
function setChartWindow(limit){chartWindow(0,0,limit);state.viewLimit=limit;state.hover=null;if(state.data)renderChart(false);}
function updateChartTooltip(){const tip=$("chart-tooltip"),i=state.hover??state.selected,b=state.data?.bars[i],pos=chartView.positions[i];if(!b||!pos){tip.hidden=true;return;}
  tip.hidden=false;tip.replaceChildren();const stamp=document.createElement('span'),price=document.createElement('strong');stamp.textContent=time(b.timestamp);price.textContent='Đóng '+number(b.close);tip.append(stamp,price);
  const scale=$("chart").clientWidth/chartView.width;tip.style.left=Math.max(8,Math.min(pos.x*scale-tip.offsetWidth/2,$("chart").clientWidth-tip.offsetWidth-8))+'px';tip.style.top=Math.max(8,pos.y*scale-65)+'px';}
function initChart(){if(chartView.svg)return;const v=chartView;v.svg=svgElement('svg',{role:'img'});v.title=svgElement('title');v.svg.append(v.title);
  const defs=svgElement('defs'),gradient=svgElement('linearGradient',{id:'price-fill',gradientUnits:'userSpaceOnUse',x1:0,y1:26,x2:0,y2:280});gradient.append(svgElement('stop',{offset:0,'stop-color':'var(--accent)','stop-opacity':.18}),svgElement('stop',{offset:1,'stop-color':'var(--accent)','stop-opacity':.015}));v.fill=gradient;defs.append(gradient);v.svg.append(defs);v.areas=svgElement('g',{'aria-hidden':'true'});v.svg.append(v.areas);
  for(let j=0;j<5;j++){const line=svgElement('line',{class:'grid'}),label=svgElement('text');v.grids.push(line);v.labels.push(label);v.svg.append(line,label);}
  v.volumeLabel=svgElement('text',{},'KHỐI LƯỢNG');v.volumeAxis=svgElement('line',{class:'grid'});v.maxVol=svgElement('text');v.gaps=svgElement('g');v.spans=svgElement('g');v.lines=svgElement('g');v.bars=svgElement('g');v.selection=svgElement('line',{class:'selection-line'});v.tickGroup=svgElement('g');v.svg.append(v.volumeLabel,v.volumeAxis,v.maxVol,v.gaps,v.spans,v.lines,v.selection,v.bars,v.tickGroup);$("chart").append(v.svg);}
function renderChart(animate=true){const data=state.data;if(!data)return;const window=chartWindow(data.bars.length,state.selected,state.viewLimit),startIndex=window.start,rows=data.bars.slice(window.start,window.end);initChart();const v=chartView,updates=[];
  const highlights=importantObservationIndices(data.observations,window.start,window.end);
  cancelAnimationFrame(state.animation);const width=Math.max(320,$("chart").clientWidth),normalHeight=width<600?360:500,height=document.body.dataset.chartExpanded==='true'?Math.max(normalHeight+80,UI.chartHeight(normalHeight)):normalHeight,end=width-78,bottom=height-140,volumeBase=height-44;
  v.width=width;v.svg.setAttribute('viewBox',`0 0 ${width} ${height}`);v.svg.dataset.chartType=state.chartType;v.svg.setAttribute('aria-label',`${state.chartType==='line'?'Đường giá đóng cửa':'Biểu đồ nến'}, ${data.symbol}, ${rows.length} nến 5 phút đã biết; dùng Nến trước và Nến sau để đọc số liệu.`);v.title.textContent=data.symbol+' · '+rows.length+' nến · '+data.source_mode;
  $("chart-line").setAttribute('aria-pressed',String(state.chartType==='line'));$("chart-candles").setAttribute('aria-pressed',String(state.chartType==='candles'));$("candle-legend").hidden=state.chartType!=='candles';$("line-legend").hidden=state.chartType!=='line';
  for(const [id,limit]of [['zoom-12',12],['zoom-24',24],['zoom-all',0]])$(id).setAttribute('aria-pressed',String(state.viewLimit===limit));
  $("chart-range").textContent=time(rows[0].timestamp)+'–'+time(rows.at(-1).timestamp)+' · '+rows.length+' nến';v.svg.dataset.startIndex=startIndex;v.svg.dataset.visibleBars=rows.length;
  const high=Math.max(...rows.map(b=>b.high)),low=Math.min(...rows.map(b=>b.low)),pad=Math.max((high-low)*.13,.04),lo=low-pad,hi=high+pad;
  const y=value=>bottom-(value-lo)/(hi-lo)*(bottom-26);let offset=0;const slots=rows.map((b,i)=>{if(i&&rows[i-1].segment_id!==b.segment_id)offset+=.5;return i+offset;});const dx=(end-18)/Math.max(slots.at(-1)+1,12),x=i=>18+(slots[i]+.5)*dx;
  const attr=(el,attrs)=>Object.entries(attrs).forEach(([k,to])=>{const old=el.getAttribute(k),from=old===null?NaN:Number(old);if(typeof to==='number'&&Number.isFinite(from)&&from!==to)updates.push({el,k,from,to});else el.setAttribute(k,String(to));});
  attr(v.fill,{y1:26,y2:bottom});
  for(let j=0;j<5;j++){const value=lo+(hi-lo)*j/4;attr(v.grids[j],{x1:18,x2:end,y1:y(value),y2:y(value)});attr(v.labels[j],{x:end+14,y:y(value)+4});v.labels[j].textContent=number(value);}
  const maxVol=Math.max(1,...rows.map(b=>b.volume));attr(v.volumeLabel,{x:18,y:height-108});attr(v.volumeAxis,{x1:18,x2:end,y1:volumeBase,y2:volumeBase});attr(v.maxVol,{x:end+14,y:volumeBase-40});v.maxVol.textContent=number(maxVol,0);
  v.positions=Array(startIndex).fill(null).concat(rows.map((b,i)=>({x:x(i),y:y(b.close)})));
  const visible=new Set(rows.map(b=>b.timestamp));for(const[key,g]of v.candles)if(!visible.has(key)){g.group.remove();g.conn.remove();g.area.remove();v.candles.delete(key);}
  v.spans.replaceChildren();if(state.mode==='models')state.backend?.events.filter(e=>e.kind==='collective').forEach(e=>{const ix=rows.map((b,i)=>[b,i]).filter(([b])=>b.segment_id===e.segment_id&&Date.parse(b.timestamp)>=Date.parse(e.event_start)&&Date.parse(b.timestamp)<=Date.parse(e.event_end)).map(([,i])=>i);if(ix.length)v.spans.append(svgElement('rect',{x:x(ix[0])-dx/2,y:16,width:x(ix.at(-1))-x(ix[0])+dx,height:bottom-16,fill:'var(--attention)',opacity:.09}));});
  v.gaps.replaceChildren();rows.forEach((b,i)=>{let g=v.candles.get(b.timestamp);if(!g){g={group:svgElement('g'),wick:svgElement('line',{class:'wick'}),body:svgElement('rect',{class:'candle'}),volume:svgElement('rect',{opacity:.75}),point:svgElement('circle',{r:3,fill:'var(--attention)'}),closePoint:svgElement('circle',{r:4,class:'close-point'}),conn:svgElement('line',{class:'price-line'}),area:svgElement('path',{class:'price-area',fill:'url(#price-fill)'}),hit:svgElement('rect',{class:'hit'}),title:svgElement('title')};g.hit.append(g.title);g.group.append(g.wick,g.body,g.volume,g.point,g.closePoint,g.hit);g.hit.addEventListener('click',()=>select(Number(g.hit.dataset.bar)));g.hit.addEventListener('pointerenter',()=>{state.hover=Number(g.hit.dataset.bar);updateChartTooltip();});g.hit.addEventListener('pointerleave',()=>{state.hover=null;updateChartTooltip();});v.areas.append(g.area);v.lines.append(g.conn);v.bars.append(g.group);v.candles.set(b.timestamp,g);}
    const color=b.close>=b.open?'var(--up)':'var(--down)',w=Math.min(dx*.56,16);g.body.setAttribute('class','candle'+(i+startIndex===state.selected?' selected':''));g.body.dataset.timestamp=b.timestamp;
    attr(g.wick,{x1:x(i),x2:x(i),y1:y(b.high),y2:y(b.low),stroke:color});attr(g.body,{x:x(i)-w/2,y:Math.min(y(b.open),y(b.close)),width:w,height:Math.max(1.8,Math.abs(y(b.open)-y(b.close))),fill:color});
    const joined=i>0&&sameSegment(rows[i-1],b);g.conn.dataset.connected=String(joined);g.conn.dataset.timestamp=b.timestamp;
    g.wick.setAttribute('visibility',state.chartType==='candles'?'visible':'hidden');g.body.setAttribute('visibility',state.chartType==='candles'?'visible':'hidden');g.conn.setAttribute('visibility',state.chartType==='line'&&joined?'visible':'hidden');
    g.area.setAttribute('visibility',state.chartType==='line'&&joined?'visible':'hidden');g.conn.style.stroke=joined&&b.close<rows[i-1].close?'var(--down)':'var(--up)';attr(g.area,{'data-base':bottom});
    if(joined)attr(g.conn,{x1:x(i-1),y1:y(rows[i-1].close),x2:x(i),y2:y(b.close)});g.closePoint.setAttribute('visibility',state.chartType==='line'&&(i+startIndex===state.selected||rows.length===1)?'visible':'hidden');attr(g.closePoint,{cx:x(i),cy:y(b.close)});
    attr(g.volume,{x:x(i)-w/2,y:volumeBase-b.volume/maxVol*46,width:w,height:b.volume/maxVol*46,fill:color});
    const flag=state.mode==='observations'?highlights.has(i+startIndex):state.backend?.events.some(e=>e.kind==='point'&&Date.parse(e.event_end)===Date.parse(b.timestamp));g.point.dataset.markerKind=state.mode;g.point.setAttribute('visibility',flag?'visible':'hidden');attr(g.point,{cx:x(i),cy:y(state.chartType==='line'?b.close:b.high)-10});
    attr(g.hit,{x:x(i)-dx/2,y:14,width:dx,height:volumeBase-12});g.hit.dataset.bar=i+startIndex;g.title.textContent=`${time(b.timestamp)} · Mở ${number(b.open)} · Cao ${number(b.high)} · Thấp ${number(b.low)} · Đóng ${number(b.close)} · Khối lượng ${number(b.volume,0)}`;
    if(i&&rows[i-1].segment_id!==b.segment_id){const mid=(x(i)+x(i-1))/2;v.gaps.append(svgElement('line',{x1:mid,x2:mid,y1:16,y2:volumeBase+1,stroke:'var(--line)','stroke-dasharray':'3 5'}));}
  });
  attr(v.selection,{x1:x(state.selected-startIndex),x2:x(state.selected-startIndex),y1:16,y2:volumeBase+1});
  let prior=-100;const tickKeys=new Set();rows.forEach((b,i)=>{const last=i===rows.length-1,candidate=i===0||last||i%8===0||rows[i-1]?.segment_id!==b.segment_id;
    if(candidate&&(last||x(i)-prior>=62)&&(last||x(rows.length-1)-x(i)>=55)){tickKeys.add(b.timestamp);let node=v.ticks.get(b.timestamp);if(!node){node=svgElement('text',{'text-anchor':'middle'});v.ticks.set(b.timestamp,node);v.tickGroup.append(node);}node.textContent=time(b.timestamp);attr(node,{x:x(i),y:height-16});prior=x(i);}});
  for(const[key,node]of v.ticks)if(!tickKeys.has(key)){node.remove();v.ticks.delete(key);}
  const duration=animate&&!matchMedia('(prefers-reduced-motion: reduce)').matches?180:0,start=performance.now();
  const paint=now=>{const t=duration?Math.min(1,(now-start)/duration):1,ease=1-Math.pow(1-t,3);updates.forEach(({el,k,from,to})=>el.setAttribute(k,String(from+(to-from)*ease)));
    for(const g of v.candles.values())if(g.conn.dataset.connected==='true'){const a=k=>g.conn.getAttribute(k),base=g.area.getAttribute('data-base');g.area.setAttribute('d',`M${a('x1')} ${base} L${a('x1')} ${a('y1')} L${a('x2')} ${a('y2')} L${a('x2')} ${base} Z`);}
    if(t<1)state.animation=requestAnimationFrame(paint);};paint(start);updateChartTooltip();
}
$("chart-line").addEventListener('click',()=>setChartType('line'));$("chart-candles").addEventListener('click',()=>setChartType('candles'));
window.addEventListener('stock-chart-layout',()=>state.data&&renderChart(false));
for(const[id,limit]of [['zoom-12',12],['zoom-24',24],['zoom-all',0]])$(id).addEventListener('click',()=>setChartWindow(limit));
$("company-logo").addEventListener('error',e=>{e.target.hidden=true;});
$("nav-overview").addEventListener('click',()=>setMode('observations'));$("nav-signals").addEventListener('click',()=>setMode('models'));
$("source").addEventListener('change',()=>{sourceControls();load();});for(const id of ['session','model'])$(id).addEventListener('change',load);
$("retry").addEventListener('click',()=>state.catalog?load():start());$("retry-model").addEventListener('click',()=>{invalidateModel();syncModel();});
$("model-retry-inline").addEventListener('click',()=>{invalidateModel();syncModel();});
$("observations-tab").addEventListener('click',()=>setMode('observations'));$("models-tab").addEventListener('click',()=>setMode('models'));
$("event-filter").addEventListener('change',()=>state.data&&renderEvents());$("timeline").addEventListener('input',e=>setFrame(Number(e.target.value)-1));
$("back").addEventListener('click',()=>setFrame(state.frame-1));$("next").addEventListener('click',()=>setFrame(state.frame+1));$("full").addEventListener('click',()=>setFrame(state.tape.frames.length-1));$("play").addEventListener('click',play);
$("select-prev").addEventListener('click',()=>select(state.selected-1));$("select-next").addEventListener('click',()=>select(state.selected+1));$("search").addEventListener('input',()=>state.data&&renderSymbols());
for(const id of ['method-info','forecast-info'])$(id).addEventListener('toggle',maybeSyncModel);
document.addEventListener('visibilitychange',()=>{if(document.hidden){stop(false);if(state.data){renderModel();renderEvents();}}else maybeSyncModel();});let measuredWidth=0;new ResizeObserver(entries=>{const w=entries[0].contentRect.width;if(w!==measuredWidth){measuredWidth=w;if(state.data)requestAnimationFrame(()=>renderChart(false));}}).observe($("chart"));
async function start(){try{state.catalog=await request('/api/catalog',{signal:AbortSignal.timeout(60000)});for(const o of $("source").options)o.disabled=!state.catalog.sources[o.value].available;if(!state.catalog.sources.synthetic.available)$("source").value='historical';sourceControls();await load();}catch(e){$("loading").hidden=true;$("error").hidden=false;$("error-copy").textContent='Không kết nối được máy chủ local: '+e.message;$("workbench").setAttribute('aria-busy','false');}}
// Move the same node rather than copying prices or changing the data/replay scope.
const compactWatchlist=matchMedia('(max-width:720px)');
function placeWatchlist(){const list=$("watchlist"),host=$(compactWatchlist.matches?'mobile-watchlist-host':'sidebar-watchlist-host');if(list.parentElement!==host){const focus=list.contains(document.activeElement)?document.activeElement:null;host.append(list);focus?.focus({preventScroll:true});}}
compactWatchlist.addEventListener('change',placeWatchlist);placeWatchlist();
start();
