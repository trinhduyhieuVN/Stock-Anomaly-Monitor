"use strict";

// The browser reads a local projection only. It cannot start a provider session.
function monitorGeometry(bars, type="line", width=900, height=360) {
  width=Math.max(300,Math.min(1600,width));const right=width-90,plot=right-28;
  height=Math.max(360,Math.min(800,height));const priceHeight=height-130,priceBottom=height-110,volumeBase=height-40,volumeHeight=Math.min(80,height*.125);
  const rows=bars.filter(b=>[b.open,b.high,b.low,b.close,b.volume].every(Number.isFinite));
  if(!rows.length)return {grid:[],prices:[],areas:[],volumes:[],labels:[]};
  const min=Math.min(...rows.map(b=>type==="line"?b.close:b.low)),max=Math.max(...rows.map(b=>type==="line"?b.close:b.high));
  const span=Math.max(max-min,Math.abs(max)*.0004,1e-6),lo=min-span*.12,hi=max+span*.12;
  const x=i=>28+(i+.5)*plot/rows.length,y=p=>20+(hi-p)/(hi-lo)*priceHeight;
  const barWidth=Math.min(14,plot/rows.length*.6),volumeMax=Math.max(...rows.map(b=>b.volume),1);
  const result={grid:[],prices:[],areas:[],volumes:[],labels:[]};
  for(let i=0;i<5;i++){const yy=20+i*priceHeight/4;result.grid.push({y:yy,label:hi-i*(hi-lo)/4,right});}
  let segment=[];
  function finish(){if(!segment.length)return;const path=segment.map(([i,b],j)=>`${j?"L":"M"}${x(i)},${y(b.close)}`).join(" ");if(segment.length>1)result.areas.push(`${path} L${x(segment.at(-1)[0])},${priceBottom} L${x(segment[0][0])},${priceBottom} Z`);segment=[];}
  rows.forEach((b,i)=>{
    const previous=rows[i-1],continuous=previous&&previous.segment_id===b.segment_id&&Date.parse(b.timestamp)-Date.parse(previous.timestamp)===300000;
    if(!continuous)finish();segment.push([i,b]);
    const color=b.close>=b.open?"var(--up)":"var(--down)";
    if(type==="candles")result.prices.push({kind:"candle",x:x(i),top:y(b.high),bottom:y(b.low),y:Math.min(y(b.open),y(b.close)),height:Math.max(1,Math.abs(y(b.open)-y(b.close))),width:barWidth,color});
    else if(continuous)result.prices.push({kind:"line",x1:x(i-1),y1:y(previous.close),x2:x(i),y2:y(b.close),color:b.close>=previous.close?"var(--up)":"var(--down)"});
    else result.prices.push({kind:"point",x:x(i),y:y(b.close),color});
    result.volumes.push({x:x(i)-barWidth/2,y:volumeBase-b.volume/volumeMax*volumeHeight,width:barWidth,height:b.volume/volumeMax*volumeHeight,color});
    if(i===0||i===rows.length-1||width>=600&&i%Math.max(1,Math.floor(rows.length/5))===0)result.labels.push({x:x(i),stamp:b.timestamp});
  });finish();return result;
}
function monitorRoute(model) {
  if(model==="full")return "/api/monitor";
  if(model==="gru6")return "/api/monitor/gru6";
  throw Error("Unknown monitor model");
}
function monitorDateTime(value) {
  const copy=typeof module!=="undefined"?require('./ui-copy.js'):window.StockMonitorUI;
  return copy.dateTime(value);
}
if(typeof module!=="undefined")module.exports={monitorGeometry,monitorRoute,monitorDateTime};

if(typeof document!=="undefined"){
  const UI=window.StockMonitorUI;
  const $=id=>document.getElementById(id),num=(v,d=2)=>Number.isFinite(v)?v.toLocaleString("vi-VN",{maximumFractionDigits:d,minimumFractionDigits:d}):"Chưa có";
  const stamp=v=>v?new Date(v).toLocaleTimeString("vi-VN",{timeZone:"Asia/Ho_Chi_Minh",hour:"2-digit",minute:"2-digit",second:"2-digit"}):"Chưa có";
  const states=UI.decisions;
  let current=null,type="line",busy=false,chartKey="",timer=null,model="full",requestId=0,activeController=null,disposed=false;
  function el(name,attrs={},text){const node=document.createElementNS("http://www.w3.org/2000/svg",name);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));if(text!==undefined)node.textContent=text;return node;}
  function draw(){const bars=current?.bars||[],width=Math.max(300,Math.round($("monitor-chart").clientWidth)),height=UI.chartHeight(360),key=JSON.stringify([type,bars,width,height]);if(key===chartKey)return;chartKey=key;const geometry=monitorGeometry(bars,type,width,height);$("monitor-svg").setAttribute("viewBox",`0 0 ${width} ${height}`);
    const grid=[],prices=[],areas=[],volumes=[],axis=[];
    geometry.grid.forEach(g=>{grid.push(el("line",{x1:28,x2:g.right,y1:g.y,y2:g.y,class:"grid"}));axis.push(el("text",{x:g.right+12,y:g.y+5},num(g.label,0)));});
    geometry.areas.forEach(d=>areas.push(el("path",{d,fill:"url(#monitor-area)"})));
    geometry.prices.forEach(p=>{if(p.kind==="line")prices.push(el("line",{x1:p.x1,y1:p.y1,x2:p.x2,y2:p.y2,stroke:p.color,"stroke-width":2.8,"stroke-linecap":"round"}));else if(p.kind==="point")prices.push(el("circle",{cx:p.x,cy:p.y,r:2,fill:p.color}));else{prices.push(el("line",{x1:p.x,x2:p.x,y1:p.top,y2:p.bottom,stroke:p.color,"stroke-width":1.4}));prices.push(el("rect",{x:p.x-p.width/2,y:p.y,width:p.width,height:p.height,fill:p.color}));}});
    geometry.volumes.forEach(v=>volumes.push(el("rect",{...v,fill:v.color,opacity:.7})));geometry.labels.forEach(l=>axis.push(el("text",{x:l.x,y:height-12,"text-anchor":"middle"},stamp(l.stamp).slice(0,5))));
    [["monitor-grid",grid],["monitor-price-layer",prices],["monitor-area-layer",type==="line"?areas:[]],["monitor-volume-layer",volumes],["monitor-axis",axis]].forEach(([id,nodes])=>$(id).replaceChildren(...nodes));
    $("monitor-svg").toggleAttribute("hidden",!bars.length);$("monitor-empty").hidden=!!bars.length;
  }
  function text(id,value){$(id).textContent=value;}
  function render(value){current=value;const mode=value.source_mode;
    text("monitor-model-rule",model==="gru6"?"GRU6 · chấm từng nến; chưa thay mô hình mặc định, không dự báo giá.":"GRU12 · phân biệt nến đơn lẻ và đoạn liên tiếp.");
    text("monitor-mode",UI.sourceName(mode));text("monitor-source-status",UI.sourceStatus(mode));
    const status=UI.monitorState(value);text("state-heading",status[0]);text("state-message",status[1]);
    const polls=UI.pollSummary(value);text("poll-count",polls);$("poll-count").hidden=!polls;
    const bars=value.bars||[],last=bars.at(-1),first=bars[0],symbol=value.symbol||last?.symbol;
    text("monitor-symbol",symbol||"Chưa có mã");text("monitor-day",UI.date(value.calendar_date||last?.session_id));
    const logos={FPT:["FPT.svg","FPT"],HPG:["HPG.png","Hòa Phát"],VCB:["VCB.png","Vietcombank"]};$("monitor-logo").hidden=!logos[symbol];if(logos[symbol]){$("monitor-logo").src="/logos/"+logos[symbol][0];$("monitor-logo").alt="Logo "+logos[symbol][1];}
    text("monitor-price",last?num(last.close):"Chưa có giá");const delta=last&&first?.open>0?(last.close/first.open-1)*100:null;text("monitor-change",Number.isFinite(delta)?`${delta>=0?"+":""}${num(delta)}%`:"");$("monitor-change").className=delta<0?"negative":"positive";
    text("monitor-bar-count",bars.length?`${bars.length} nến đã đóng`:"Chưa có nến");
    text("monitor-latest",stamp(last?.timestamp));text("monitor-received",monitorDateTime(value.received_at_utc));const completed=model==="gru6"?value.score_finished_at_utc:value.process_finished_at_utc||value.score_finished_at_utc;text("monitor-scored",completed?monitorDateTime(completed):"Chưa chấm");
    text("monitor-duration",Number.isFinite(value.score_duration_seconds)?num(value.score_duration_seconds)+" s":"Chưa đo");
    text("monitor-latency",mode==="OFFLINE_RECEIPT_REHEARSAL"?"Không đo từ dữ liệu cũ":mode==="RECEIVED_MARKET_DATA"&&Number.isFinite(value.actual_receive_to_score_seconds)?num(value.actual_receive_to_score_seconds)+" s":"Chưa đo");
    text("monitor-delay",Number.isFinite(value.provider_delay_seconds)?num(value.provider_delay_seconds,0)+" s"+(value.delay_basis==="offline_hypothesis_not_current_measurement"?" · giả định khi chấm lại":""):"Chưa đo");text("monitor-gaps",Number.isFinite(value.readiness?.gap_count)?String(value.readiness.gap_count):"Chưa kiểm tra");
    text("monitor-artifact",value.ai_manifest_sha256||value.manifest_sha256?"Đã kiểm tra":"Chưa kiểm tra");
    const events=value.events||[];$("monitor-events").replaceChildren(...events.slice(-48).reverse().map(e=>{const row=document.createElement("tr");[stamp(e.target_time).slice(0,5),states[e.emission.state]||"Chưa đủ điều kiện",`${num(e.emission.joint_gain)} / ${num(e.emission.threshold)}`,monitorDateTime(e.available_at)].forEach(content=>{const cell=document.createElement("td");cell.textContent=content;row.append(cell);});return row;}));$("monitor-events-empty").hidden=!!events.length;
    $("monitor-table-wrap").hidden=!events.length;$("monitor-forecast-info").hidden=model==="gru6";
    const forecasts=(value.forecasts||[]).slice(-2),outcomes=(value.outcomes||[]).filter(o=>o.kind==="outcome");$("monitor-forecasts").replaceChildren(...(forecasts.length?forecasts.map(f=>{const p=document.createElement("p"),o=outcomes.find(o=>o.forecast_id===f.forecast_id);p.textContent=`${f.horizon_bars*5} phút · ${f.algorithm==="zero_return"?"Mốc lợi suất 0":"Ridge"}: ${f.horizon_status==="censored_boundary"?"Qua nghỉ phiên, không đánh giá":f.horizon_status==="censored_late_creation"?"Tạo trễ, không đánh giá":o?`Đã nhận kết quả · biến động ${num(o.realized_return*100)}%`:`Dự báo ${num(f.prediction*100)}% · chờ nến kết quả`}`;return p;}):[Object.assign(document.createElement("p"),{textContent:"Chưa có dự báo cho nến này."})]));draw();
  }
  function schedule(){clearTimeout(timer);if(!disposed)timer=setTimeout(()=>{if(document.hidden)schedule();else refresh();},5000);}
  async function refresh(changed=false){if(disposed||busy&&!changed)return;clearTimeout(timer);activeController?.abort();const id=++requestId,requestedModel=model;busy=true;$("refresh-monitor").disabled=true;const controller=new AbortController();activeController=controller;const timeout=setTimeout(()=>controller.abort(),10000);
    if(changed)render({state:"LOADING",source_mode:"NO_SESSION",bars:[],events:[],forecasts:[],message:"Đang đọc snapshot riêng. Không chấm lại hoặc mở nguồn dữ liệu."});
    try{const response=await fetch(monitorRoute(requestedModel),{signal:controller.signal,cache:"no-store"});if(!response.ok)throw Error("Trạng thái local không qua kiểm tra");const value=await response.json();if(!disposed&&id===requestId&&requestedModel===model)render(value);}
    catch(error){if(!disposed&&id===requestId){text("state-heading","Không kiểm tra được dữ liệu");text("state-message",current?.bars?.length?"Đang giữ dữ liệu lần kiểm tra trước. Chọn Làm mới để thử lại.":"Chọn Làm mới để thử lại; chưa có kết quả được xác minh.");text("monitor-artifact","Chưa xác minh lần đọc này");}}
    finally{clearTimeout(timeout);if(!disposed&&id===requestId){busy=false;activeController=null;$("refresh-monitor").disabled=false;schedule();}}
  }
  [...$("monitor-model").options].forEach(option=>{option.textContent=UI.modelLabel(option.value==="full"?"ai_context":option.value);});
  $("monitor-model").value="full";$("monitor-model").addEventListener("change",()=>{model=$("monitor-model").value;refresh(true);});
  $("refresh-monitor").addEventListener("click",()=>{clearTimeout(timer);refresh();});
  ["line","candles"].forEach(choice=>$("monitor-"+choice).addEventListener("click",()=>{type=choice;["line","candles"].forEach(c=>$("monitor-"+c).setAttribute("aria-pressed",String(c===type)));draw();}));
  const observer=new ResizeObserver(()=>draw());observer.observe($("monitor-chart"));
  window.addEventListener('stock-chart-layout',draw);
  window.addEventListener("pagehide",()=>{disposed=true;requestId++;clearTimeout(timer);activeController?.abort();observer.disconnect();});
  window.addEventListener("pageshow",event=>{if(event.persisted){disposed=false;busy=false;observer.observe($("monitor-chart"));refresh();}});refresh();
}
