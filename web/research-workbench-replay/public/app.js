"use strict";
const $ = id => document.getElementById(id);
const state = {catalog:null,tape:null,data:null,frame:0,selected:0,playing:false,timer:null,
  generation:0,abort:null,modelAbort:null,modelCache:new Map(),backend:null,modelKey:null,
  modelError:null,modelPending:false,modelTimer:null,mode:"observations",reviewId:null,saving:false,paint:0,animation:0};
const number = (v,d=2) => v==null||!Number.isFinite(Number(v)) ? "Chưa có" : Number(v).toLocaleString("vi-VN",{minimumFractionDigits:d,maximumFractionDigits:d});
const time = stamp => new Date(stamp).toLocaleTimeString("vi-VN",{timeZone:"Asia/Ho_Chi_Minh",hour:"2-digit",minute:"2-digit"});
const knownTime = stamp => new Date(stamp).toLocaleTimeString("vi-VN",{timeZone:"Asia/Ho_Chi_Minh",hour:"2-digit",minute:"2-digit",second:"2-digit"});
const names = {FPT:"Công nghệ",HPG:"Thép",VCB:"Ngân hàng"};
const kinds = {point:"Điểm cần xem",collective:"Đoạn cần xem",normal:"Không vượt ngưỡng",insufficient_data:"Chưa đủ dữ liệu"};
const labels = {noteworthy:"Đáng chú ý",ordinary:"Bình thường",uncertain:"Chưa rõ",data_issue:"Lỗi dữ liệu"};
function paragraphs(target,lines){target.replaceChildren();lines.filter(Boolean).forEach(line=>{const p=document.createElement("p");p.textContent=line;target.append(p);});}
function options(target,rows,value){target.replaceChildren();rows.forEach(([name,v])=>{const o=document.createElement("option");o.value=v;o.textContent=name;target.append(o);});if(rows.some(r=>r[1]===value))target.value=value;}
function query(scope){const p=new URLSearchParams();Object.entries(scope).forEach(([k,v])=>{if(v!=null)p.set(k,String(v));});return p.toString();}
function tapeScope(){return {source:$("source").value,symbol:state.symbol||"FPT",session:$("session").value,model:$("model").value};}
function scope(){const f=state.tape.frames[state.frame];return {...tapeScope(),replay:state.frame<state.tape.frames.length-1,step:f.step};}
function scopeKey(){return query(scope());}
function stop(sync=true){clearTimeout(state.timer);state.timer=null;state.playing=false;$("play").textContent="Phát Replay";if(sync)maybeSyncModel();}
async function request(path,opts={}){const response=await fetch(path,opts);const data=await response.json();if(!response.ok)throw Error(data.error||"Máy chủ local chưa sẵn sàng.");return data;}
function sourceControls(){const source=state.catalog.sources[$("source").value];options($("session"),source.sessions.map(v=>[v,v]),$("source").value==="synthetic"?"2025-02-03":source.sessions.at(-1));options($("model"),Object.entries(source.models));state.symbol=source.symbols[0]||"FPT";}
function clearSelection(){state.reviewId=null;$("review-reason").value="";$("review-status").textContent="";$("reviews").replaceChildren();}
function invalidateModel(){clearTimeout(state.modelTimer);state.modelTimer=null;state.modelAbort?.abort();state.modelAbort=null;state.modelKey=null;state.backend=null;state.modelError=null;state.modelPending=false;}
async function load(){
  stop(false);clearSelection();invalidateModel();state.modelCache.clear();state.tape=null;state.data=null;
  const generation=++state.generation;state.abort?.abort();state.abort=new AbortController();
  $("content").hidden=true;$("error").hidden=true;$("loading").hidden=false;$("workbench").setAttribute("aria-busy","true");
  $("source-status").textContent="Đang kiểm tra nguồn và phiên đang chọn…";
  try{const tape=await request('/api/replay-tape?'+query(tapeScope()),{signal:AbortSignal.any([state.abort.signal,AbortSignal.timeout(60000)])});
    if(generation!==state.generation)return;state.tape=tape;state.frame=tape.frames.length-1;state.selected=tape.frames.at(-1).bars.length-1;
    if(tape.live!==false||tape.model_scores_included!==false||!tape.frames.length)throw Error("Tape offline không hợp lệ");
    $("content").hidden=false;applyFrame(false);renderSymbols();maybeSyncModel();
  }catch(e){if(generation!==state.generation||e.name==="AbortError")return;state.tape=null;state.data=null;$("content").hidden=true;$("error-copy").textContent="Không đọc được dữ liệu: "+e.message;$("error").hidden=false;$("source-status").textContent="Dữ liệu chưa sẵn sàng. Không giữ biểu đồ nguồn trước.";}
  finally{if(generation===state.generation){$("loading").hidden=true;$("workbench").setAttribute("aria-busy","false");}}
}
// Same descriptive rules as the first prototype. No model event IDs or scores.
function observations(rows){return rows.map((b,i)=>{const prior=rows.slice(Math.max(0,i-12),i).map(r=>r.volume).sort((a,b)=>a-b),n=prior.length;
  const median=n?(prior[Math.floor((n-1)/2)]+prior[Math.floor(n/2)])/2:null;
  const change=i&&rows[i-1].close>0?(b.close/rows[i-1].close-1)*100:null,ratio=median>0?b.volume/median:null;
  const priceEvent=change!==null&&Math.abs(change)>=.6,volumeEvent=ratio!==null&&ratio>=2;
  return {index:i,change,ratio,reference:n,priceEvent,volumeEvent,
    label:priceEvent?(volumeEvent?"Biến động giá + volume":"Biến động giá"):"Khối lượng tăng",
    value:priceEvent?`${change>=0?'+':''}${number(change)}%`:number(ratio,1)+"×"};});}
function modelWanted(){return state.mode==="models"||['method-info','forecast-info','review-section'].some(id=>$(id).open);}
function maybeSyncModel(){if(state.data&&modelWanted()&&!state.playing){clearTimeout(state.modelTimer);state.modelTimer=setTimeout(()=>{state.modelTimer=null;if(state.data&&modelWanted()&&!state.playing)syncModel();},150);}}
async function syncModel(){
  if(!state.data||state.playing)return;
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
  const status=state.playing?"Mô hình sẽ đồng bộ khi tạm dừng.":state.modelError?"Chưa đọc được mô hình: "+state.modelError:state.modelPending||state.modelTimer?"Đang đọc kết quả tại mốc đang xem…":state.backend?"Đã đọc kết quả đến "+time(state.backend.as_of):"Mở phần này để đọc kết quả tại mốc đang xem.";
  $("model-status").textContent=status;
  $("model-overview").hidden=state.mode!=="models";
  const events=state.backend?.events||[],scored=new Set(events.filter(e=>e.score!=null).map(e=>Date.parse(e.decision_bar_end||e.event_end))).size;
  const unavailable=new Set(events.filter(e=>e.kind==='insufficient_data').map(e=>Date.parse(e.decision_bar_end||e.event_end))).size;
  const flagged=events.filter(e=>['point','collective'].includes(e.kind)).length;
  const episodeCopy=state.backend?.model_metadata?` · ${state.backend.episodes.length} hồ sơ (${flagged} emission được đánh dấu)`: ` · ${flagged} cảnh báo`;
  $("model-overview").textContent=state.backend?`${$("model").selectedOptions[0]?.textContent||data.model} · ${scored}/${data.bars.length} nến có score${episodeCopy} · ${unavailable} nến chưa đủ dữ liệu · đến ${time(state.backend.as_of)}.`:status;
  $("retry-model").hidden=!state.modelError;
  $("model-retry-inline").hidden=!state.modelError||state.mode!=='models';
  paragraphs($("method-copy"),detail?.method||["Chưa có score đã đồng bộ cho nến này."]);
  $("score").textContent=event?.score==null?(event?.kind==="insufficient_data"?"Cửa sổ chưa đủ dữ liệu":state.backend?"Không có score cho kết quả này":"Chờ đồng bộ"):number(event.score,5);
  $("threshold").textContent=event?.threshold==null?"Không áp dụng / chưa có":number(event.threshold,5);
  paragraphs($("data-copy"),[data.source==='historical'?'Nguồn thật · local-only · quyền nguồn và đơn vị chưa xác nhận. Không có receipt từng nến.':'Giả lập · thời gian logic, không phải độ trễ provider.',`Manifest: ${data.provenance.manifest_sha256}`]);
  $("forecast-info").hidden=data.source!=="synthetic"&&!state.backend?.model_metadata;
  $("forecast-basis").textContent=data.source==='synthetic'?'Baseline lợi suất 0 · dữ liệu giả lập.':state.backend?.model_metadata?.forecast_algorithm==='zero_return'?'Baseline lợi suất 0 được giữ vì Ridge chưa vượt tiêu chí validation.':'Ridge h1/h3 · thí nghiệm dữ liệu lịch sử.';
  paragraphs($("forecasts"),(detail?.forecasts||[]).map(f=>`${f.horizon_bars} nến · đích ${time(f.target_bar_end)} · dự báo ${number(100*f.prediction,4)}% · ${f.status==='mature'?'Đã có kết quả':f.status==='pending'?'Đang chờ kết quả':f.status==='censored_boundary'?'Qua ranh giới phiên; không đối chiếu':'Thiếu nến; chưa đối chiếu'}${f.realized_return==null?'':'; thực hiện '+number(100*f.realized_return,4)+'%; sai lệch '+number(100*f.residual,4)+'%'}.`));
  if(!detail?.forecasts?.length)paragraphs($("forecasts"),["Chưa có mốc đối chiếu đã đồng bộ cho nến này."]);
  $("version-info").hidden=!detail?.initial_event;
  paragraphs($("event-versions"),(detail?.versions||[]).map(v=>`v${v.version} · ${time(v.available_at)} · ${v.stage==='initial'?'Ban đầu: '+kinds[v.kind]:'Đã đủ '+v.post_evidence.horizon_bars+' nến sau quyết định; giữ nguyên đánh giá ban đầu.'}`));
  $("save-review").disabled=data.replay||state.playing||!event||state.saving;
  $("review-status").textContent=data.replay||state.playing?"Xem cả phiên để ghi nhận xét local.":!event?"Chưa có sự kiện mô hình đã đồng bộ ở nến này.":"";
  const reviews=event?state.backend.reviews.filter(r=>r.event_id===event.event_id&&r.event_version===event.version):[];
  paragraphs($("reviews"),reviews.map(r=>`${r.reviewer} · ${r.reviewer_kind==='ai_assisted'?'Có AI hỗ trợ':'Người dùng'} · ${labels[r.label]}: ${r.reason}`));
}
function renderDetails(){const data=state.data,b=data.bars[state.selected],o=data.observations[state.selected];if(!b)return;
  $("selection-time").textContent=`${data.symbol} · ${time(b.timestamp)}`;
  $("selection-kind").textContent=state.mode==='models'?(backendEvent()?kinds[backendEvent().kind]:"Chưa có tín hiệu đồng bộ"):(o.priceEvent||o.volumeEvent?o.label:"Chưa vượt quy tắc quan sát");
  for(const k of ['open','high','low','close'])$(k).textContent=number(b[k]);$("volume").textContent=number(b.volume,0);
  $("return").textContent=o.change==null?"Chưa có nến trước":`${o.change>=0?'+':''}${number(o.change)}%`;
  $("volume-ratio").textContent=o.ratio==null?"Chưa có nền hợp lệ":number(o.ratio,1)+"×";
  paragraphs($("description"),[o.change==null?"Nến đầu tiên trong phần đang xem.":`Giá đóng ${o.change>=0?'tăng':'giảm'} ${number(Math.abs(o.change))}% so với nến có dữ liệu trước.`,o.ratio==null?"Chưa có trung vị volume dương để so sánh.":`Khối lượng bằng ${number(o.ratio,1)}× trung vị ${o.reference} nến trước có dữ liệu.`]);
  $("read-selection").textContent="Đọc nến "+time(b.timestamp)+" đang chọn";
  $("select-prev").disabled=state.selected===0;$("select-next").disabled=state.selected===data.bars.length-1;renderModel();
}
function select(index){if(!state.data)return;clearSelection();state.selected=Math.max(0,Math.min(index,state.data.bars.length-1));renderChart(false);renderDetails();renderEvents();}
function renderEvents(){const data=state.data,target=$("events"),filter=$("event-filter").value;target.replaceChildren();
  const entries=state.mode==='observations'?data.observations.filter(o=>(o.priceEvent||o.volumeEvent)&&(filter==='all'||filter==='price'&&o.priceEvent||filter==='volume'&&o.volumeEvent)).map(o=>({index:o.index,label:o.label,value:o.value,id:String(o.index)})):
    (state.backend?.events||[]).slice().reverse().filter(e=>filter==='all'||filter==='flagged'&&['point','collective'].includes(e.kind)||e.kind===filter).map(e=>({index:data.bars.findIndex(b=>Date.parse(b.timestamp)===Date.parse(e.decision_bar_end||e.event_end)),label:kinds[e.kind],value:e.score!=null?`Score ${number(e.score,5)}${e.version>1?' · v'+e.version:''}`:e.kind==='insufficient_data'?'Chưa đủ cửa sổ':e.stage==='initial'?'Ban đầu':`Có đối chiếu · v${e.version}`,id:e.event_id}));
  entries.filter(e=>e.index>=0).forEach(e=>{const b=document.createElement('button');b.className='event';b.dataset.eventId=e.id;b.setAttribute('aria-pressed',String(e.index===state.selected));
    for(const[tag,txt]of[['time',time(data.bars[e.index].timestamp)],['strong',e.label],['span',e.value]]){const el=document.createElement(tag);el.textContent=txt;if(tag==='span'&&state.mode==='observations')el.className='observation-value';b.append(el);}
    b.addEventListener('click',()=>{select(e.index);target.querySelector('[data-event-id="'+e.id+'"]')?.focus();});target.append(b);});
  $("no-events").hidden=target.childElementCount>0;$("no-events").textContent=state.mode==='models'?(state.modelError?'Chưa đọc được mô hình; có thể thử lại bên dưới.':!state.backend?state.playing?'Tạm dừng để đọc tín hiệu tại mốc đang xem.':'Đang chờ đồng bộ kết quả…':filter==='flagged'?'Không có nến vượt ngưỡng trong phần đang xem. Chọn Tất cả nến để đọc score.':'Không có kết quả theo bộ lọc này.'):'Không có điểm theo bộ lọc này.';
}
function setMode(mode){state.mode=mode;$("observations-tab").setAttribute('aria-pressed',String(mode==='observations'));$("models-tab").setAttribute('aria-pressed',String(mode==='models'));$("rule-copy").hidden=mode!=='observations';$("events-title").textContent=mode==='observations'?'Biến động quan sát':'Tín hiệu mô hình';$("mark-label").textContent=$("events-title").textContent;
  options($("event-filter"),mode==='observations'?[['Tất cả loại','all'],['Biến động giá','price'],['Khối lượng tăng','volume']]:[['Tất cả nến','all'],['Được đánh dấu','flagged'],['Điểm đơn lẻ','point'],['Đoạn liên tiếp','collective'],['Không vượt ngưỡng','normal'],['Chưa đủ dữ liệu','insufficient_data']]);
  if(mode==='models')$("method-info").open=true;
  if(state.data){renderEvents();renderDetails();renderChart(false);maybeSyncModel();}}
function renderSymbols(){const data=state.data,target=$("symbols"),search=$("search").value.trim().toUpperCase();target.replaceChildren();const symbols=state.catalog.sources[data.source].symbols;$("symbol-count").textContent=symbols.length+" mã";
  symbols.filter(s=>s.includes(search)).forEach(s=>{const button=document.createElement('button');button.className='symbol';button.dataset.symbol=s;button.setAttribute('aria-pressed',String(s===data.symbol));button.setAttribute('aria-label','Xem mã '+s);
    const top=document.createElement('span');top.className='symbol-top';const strong=document.createElement('strong');strong.textContent=s;top.append(strong);
    const bottom=document.createElement('span');bottom.className='symbol-bottom';const name=document.createElement('span');name.textContent=names[s]||s;bottom.append(name);
    if(s===data.symbol){const value=document.createElement('span');value.className='symbol-price';top.append(value);const delta=document.createElement('span');delta.className='symbol-delta';bottom.append(delta);}
    button.append(top,bottom);button.addEventListener('click',async()=>{state.symbol=s;await load();if(state.data?.symbol===s)target.querySelector('[data-symbol="'+s+'"]')?.focus();});target.append(button);});$("no-symbols").hidden=target.childElementCount>0;updateSymbolQuote();}
function updateSymbolQuote(){const rows=state.data?.bars;if(!rows?.length)return;const p=$("symbols").querySelector('.symbol-price'),d=$("symbols").querySelector('.symbol-delta'),change=(rows.at(-1).close/rows[0].open-1)*100;if(p)p.textContent=number(rows.at(-1).close);if(d){d.textContent=(change>=0?'+':'')+number(change)+'%';d.className='symbol-delta '+(change>=0?'positive':'negative');}}
function applyFrame(animate=true){const tape=state.tape,f=tape.frames[state.frame];state.data={...tape,...f,replay:state.frame<tape.frames.length-1};state.data.observations=observations(f.bars);state.selected=Math.min(state.selected,Math.max(0,f.bars.length-1));
  const last=f.bars.at(-1),first=f.bars[0],change=first.open?(last.close/first.open-1)*100:null;
  $("chart-heading").textContent=tape.symbol;$("instrument-name").textContent=tape.session+" · Nến 5 phút";$("last-price").textContent=number(last.close);$("day-change").textContent=change==null?'':(change>=0?'+':'')+number(change)+'%';$("day-change").className=change>=0?'positive':'negative';
  $("known-time").textContent='Đã biết đến '+knownTime(f.as_of)+(tape.source==='synthetic'?' (giả lập)':' (hồi cứu)');$("unit").textContent=tape.source==='synthetic'?'Đơn vị giả lập':'Đơn vị nguồn';
  $("source-status").textContent=tape.source==='synthetic'?'Giả lập · Replay offline · Không LIVE':'Lịch sử thật · Local-only · Quyền nguồn / đơn vị chưa xác nhận · Không LIVE';
  $("timeline").max=tape.frames.length;$("timeline").value=state.frame+1;$("timeline").setAttribute('aria-valuetext',f.bars.length+' nến, đã biết đến '+time(f.as_of));$("bar-count").textContent=f.bars.length+' / '+tape.frames.at(-1).bars.length+' nến';
  $("back").disabled=state.frame===0;$("next").disabled=state.frame===tape.frames.length-1;$("full").disabled=state.frame===tape.frames.length-1;
  for(const kind of ['bars','events','reviews'])$("export-"+kind).href='/api/export?'+scopeKey()+'&kind='+kind;
  updateSymbolQuote();renderChart(animate);renderDetails();renderEvents();
}
function setFrame(index,{playing=false}={}){if(!state.tape)return;if(!playing)stop(false);invalidateModel();clearSelection();state.frame=Math.max(0,Math.min(Number(index),state.tape.frames.length-1));state.selected=state.tape.frames[state.frame].bars.length-1;
  cancelAnimationFrame(state.paint);state.paint=requestAnimationFrame(()=>{applyFrame(true);if(!playing)maybeSyncModel();});}
function play(){if(!state.tape)return;if(state.playing){stop();renderModel();return;}state.playing=true;invalidateModel();$("play").textContent='Tạm dừng';if(state.frame===state.tape.frames.length-1)setFrame(0,{playing:true});else renderModel();
  const tick=()=>{if(!state.playing)return;if(state.frame>=state.tape.frames.length-1){stop();return;}setFrame(state.frame+1,{playing:true});if(state.frame>=state.tape.frames.length-1){state.playing=false;$("play").textContent='Phát Replay';requestAnimationFrame(maybeSyncModel);}else state.timer=setTimeout(tick,650);};state.timer=setTimeout(tick,650);
}
// Persistent SVG and keyed candle nodes. Geometry updates are coalesced in rAF;
// no chart teardown, fetch, busy overlay or cold-load animation during Replay.
const svgElement=(name,attrs={},text)=>{const el=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text!==undefined)el.textContent=text;return el;};
const chartView={svg:null,candles:new Map(),ticks:new Map(),grids:[],labels:[]};
function initChart(){if(chartView.svg)return;const v=chartView;v.svg=svgElement('svg',{role:'img'});v.title=svgElement('title');v.svg.append(v.title);
  for(let j=0;j<5;j++){const line=svgElement('line',{class:'grid'}),label=svgElement('text');v.grids.push(line);v.labels.push(label);v.svg.append(line,label);}
  v.volumeLabel=svgElement('text',{},'KHỐI LƯỢNG');v.volumeAxis=svgElement('line',{class:'grid'});v.maxVol=svgElement('text');v.gaps=svgElement('g');v.spans=svgElement('g');v.bars=svgElement('g');v.selection=svgElement('line',{class:'selection-line'});v.tickGroup=svgElement('g');v.svg.append(v.volumeLabel,v.volumeAxis,v.maxVol,v.gaps,v.spans,v.selection,v.bars,v.tickGroup);$("chart").append(v.svg);}
function renderChart(animate=true){const data=state.data;if(!data)return;const rows=data.bars;initChart();const v=chartView,updates=[];
  cancelAnimationFrame(state.animation);const width=Math.max(320,$("chart").clientWidth),height=width<600?340:420,end=width-78,bottom=height-140,volumeBase=height-44;
  v.svg.setAttribute('viewBox',`0 0 ${width} ${height}`);v.svg.setAttribute('aria-label',`${data.symbol}, ${rows.length} nến 5 phút đã biết; dùng Nến trước và Nến sau để đọc số liệu.`);v.title.textContent=data.symbol+' · '+rows.length+' nến · '+data.source_mode;
  const high=Math.max(...rows.map(b=>b.high)),low=Math.min(...rows.map(b=>b.low)),pad=Math.max((high-low)*.13,.04),lo=low-pad,hi=high+pad;
  const y=value=>bottom-(value-lo)/(hi-lo)*(bottom-26);let offset=0;const slots=rows.map((b,i)=>{if(i&&rows[i-1].segment_id!==b.segment_id)offset+=2;return i+offset;});const dx=(end-18)/Math.max(slots.at(-1)+1,12),x=i=>18+(slots[i]+.5)*dx;
  const attr=(el,attrs)=>Object.entries(attrs).forEach(([k,to])=>{const old=el.getAttribute(k),from=old===null?NaN:Number(old);if(typeof to==='number'&&Number.isFinite(from)&&from!==to)updates.push({el,k,from,to});else el.setAttribute(k,String(to));});
  for(let j=0;j<5;j++){const value=lo+(hi-lo)*j/4;attr(v.grids[j],{x1:18,x2:end,y1:y(value),y2:y(value)});attr(v.labels[j],{x:end+14,y:y(value)+4});v.labels[j].textContent=number(value);}
  const maxVol=Math.max(1,...rows.map(b=>b.volume));attr(v.volumeLabel,{x:18,y:height-108});attr(v.volumeAxis,{x1:18,x2:end,y1:volumeBase,y2:volumeBase});attr(v.maxVol,{x:end+14,y:volumeBase-40});v.maxVol.textContent=number(maxVol,0);
  const visible=new Set(rows.map(b=>b.timestamp));for(const[key,g]of v.candles)if(!visible.has(key)){g.group.remove();v.candles.delete(key);}
  v.spans.replaceChildren();if(state.mode==='models')state.backend?.events.filter(e=>e.kind==='collective').forEach(e=>{const ix=rows.map((b,i)=>[b,i]).filter(([b])=>b.segment_id===e.segment_id&&Date.parse(b.timestamp)>=Date.parse(e.event_start)&&Date.parse(b.timestamp)<=Date.parse(e.event_end)).map(([,i])=>i);if(ix.length)v.spans.append(svgElement('rect',{x:x(ix[0])-dx/2,y:16,width:x(ix.at(-1))-x(ix[0])+dx,height:bottom-16,fill:'var(--attention)',opacity:.09}));});
  v.gaps.replaceChildren();rows.forEach((b,i)=>{let g=v.candles.get(b.timestamp);if(!g){g={group:svgElement('g'),wick:svgElement('line',{class:'wick'}),body:svgElement('rect',{class:'candle'}),volume:svgElement('rect',{opacity:.5}),point:svgElement('circle',{r:3,fill:'var(--attention)'}),hit:svgElement('rect',{class:'hit'}),title:svgElement('title')};g.hit.append(g.title);g.group.append(g.wick,g.body,g.volume,g.point,g.hit);g.hit.addEventListener('click',()=>select(Number(g.hit.dataset.bar)));v.bars.append(g.group);v.candles.set(b.timestamp,g);}
    const color=b.close>=b.open?'var(--up)':'var(--down)',w=Math.min(dx*.56,16);g.body.setAttribute('class','candle'+(i===state.selected?' selected':''));g.body.dataset.timestamp=b.timestamp;
    attr(g.wick,{x1:x(i),x2:x(i),y1:y(b.high),y2:y(b.low),stroke:color});attr(g.body,{x:x(i)-w/2,y:Math.min(y(b.open),y(b.close)),width:w,height:Math.max(1.8,Math.abs(y(b.open)-y(b.close))),fill:color});
    attr(g.volume,{x:x(i)-w/2,y:volumeBase-b.volume/maxVol*46,width:w,height:b.volume/maxVol*46,fill:color});
    const o=data.observations[i],flag=state.mode==='observations'?(o.priceEvent||o.volumeEvent):state.backend?.events.some(e=>e.kind==='point'&&Date.parse(e.event_end)===Date.parse(b.timestamp));g.point.setAttribute('visibility',flag?'visible':'hidden');attr(g.point,{cx:x(i),cy:y(b.high)-10});
    attr(g.hit,{x:x(i)-dx/2,y:14,width:dx,height:volumeBase-12});g.hit.dataset.bar=i;g.title.textContent=`${time(b.timestamp)} · Mở ${number(b.open)} · Đóng ${number(b.close)} · Volume ${number(b.volume,0)}`;
    if(i&&rows[i-1].segment_id!==b.segment_id){const mid=(x(i)+x(i-1))/2;v.gaps.append(svgElement('line',{x1:mid,x2:mid,y1:16,y2:volumeBase+1,stroke:'var(--line)','stroke-dasharray':'3 5'}));}
  });
  attr(v.selection,{x1:x(state.selected),x2:x(state.selected),y1:16,y2:volumeBase+1});
  let prior=-100;const tickKeys=new Set();rows.forEach((b,i)=>{const last=i===rows.length-1,candidate=i===0||last||i%8===0||rows[i-1]?.segment_id!==b.segment_id;
    if(candidate&&(last||x(i)-prior>=62)&&(last||x(rows.length-1)-x(i)>=55)){tickKeys.add(b.timestamp);let node=v.ticks.get(b.timestamp);if(!node){node=svgElement('text',{'text-anchor':'middle'});v.ticks.set(b.timestamp,node);v.tickGroup.append(node);}node.textContent=time(b.timestamp);attr(node,{x:x(i),y:height-16});prior=x(i);}});
  for(const[key,node]of v.ticks)if(!tickKeys.has(key)){node.remove();v.ticks.delete(key);}
  const duration=animate&&!matchMedia('(prefers-reduced-motion: reduce)').matches?180:0,start=performance.now();
  const paint=now=>{const t=duration?Math.min(1,(now-start)/duration):1,ease=1-Math.pow(1-t,3);updates.forEach(({el,k,from,to})=>el.setAttribute(k,String(from+(to-from)*ease)));if(t<1)state.animation=requestAnimationFrame(paint);};paint(start);
}
$("source").addEventListener('change',()=>{sourceControls();load();});for(const id of ['session','model'])$(id).addEventListener('change',load);
$("retry").addEventListener('click',()=>state.catalog?load():start());$("retry-model").addEventListener('click',()=>{invalidateModel();syncModel();});
$("model-retry-inline").addEventListener('click',()=>{invalidateModel();syncModel();});
$("observations-tab").addEventListener('click',()=>setMode('observations'));$("models-tab").addEventListener('click',()=>setMode('models'));
$("event-filter").addEventListener('change',()=>state.data&&renderEvents());$("timeline").addEventListener('input',e=>setFrame(Number(e.target.value)-1));
$("back").addEventListener('click',()=>setFrame(state.frame-1));$("next").addEventListener('click',()=>setFrame(state.frame+1));$("full").addEventListener('click',()=>setFrame(state.tape.frames.length-1));$("play").addEventListener('click',play);
$("select-prev").addEventListener('click',()=>select(state.selected-1));$("select-next").addEventListener('click',()=>select(state.selected+1));$("search").addEventListener('input',()=>state.data&&renderSymbols());
for(const id of ['method-info','forecast-info','review-section'])$(id).addEventListener('toggle',maybeSyncModel);
$("review-form").addEventListener('submit',async e=>{e.preventDefault();const event=backendEvent();if(!event||state.data.replay||state.playing||state.saving)return;
  const generation=state.generation,key=scopeKey();state.saving=true;$("save-review").disabled=true;state.reviewId??=crypto.randomUUID();const body={scope:scope(),review:{review_id:state.reviewId,event_id:event.event_id,event_version:event.version,reviewer:$("reviewer").value.trim(),reviewer_kind:$("reviewer-kind").value,label:$("review-label").value,reason:$("review-reason").value.trim(),reference_url:""}};
  try{const saved=await request('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Workbench-Token':state.catalog.csrf_token},body:JSON.stringify(body),signal:AbortSignal.timeout(60000)});state.modelCache.delete(key);
    if(generation!==state.generation||!state.data||key!==scopeKey())return;state.backend.reviews.push(saved);state.reviewId=null;renderModel();$("review-status").textContent='Đã lưu nhận xét local.';
  }catch(err){if(generation===state.generation&&state.data&&key===scopeKey())$("review-status").textContent='Chưa lưu: '+err.message;}
  finally{state.saving=false;if(state.data)$("save-review").disabled=state.data.replay||state.playing||!backendEvent();}
});
document.addEventListener('visibilitychange',()=>{if(document.hidden){stop(false);if(state.data){renderModel();renderEvents();}}else maybeSyncModel();});let measuredWidth=0;new ResizeObserver(entries=>{const w=entries[0].contentRect.width;if(w!==measuredWidth){measuredWidth=w;if(state.data)requestAnimationFrame(()=>renderChart(false));}}).observe($("chart"));
async function start(){try{state.catalog=await request('/api/catalog',{signal:AbortSignal.timeout(60000)});for(const o of $("source").options)o.disabled=!state.catalog.sources[o.value].available;if(!state.catalog.sources.synthetic.available)$("source").value='historical';sourceControls();await load();}catch(e){$("loading").hidden=true;$("error").hidden=false;$("error-copy").textContent='Không kết nối được máy chủ local: '+e.message;$("workbench").setAttribute('aria-busy','false');}}
start();
