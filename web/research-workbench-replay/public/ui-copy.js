"use strict";

// Presentation only. IDs, model outputs and source evidence are never rewritten.
const StockMonitorUI=(()=>{
  const modelNames=Object.freeze({robust_zscore:"Robust z-score",isolation_forest:"Isolation Forest",lstm:"LSTM",
    v2:"Điểm / đoạn · giả lập",ai_context:"GRU12 · điểm / đoạn",ai_no_context:"GRU12 · không bối cảnh bổ sung",
    ai_point_only:"GRU12 · chỉ từng nến",ai_zero_forecast:"GRU12 · dự báo mốc 0",gru6:"GRU6 · thử nghiệm từng nến"});
  const decisions=Object.freeze({point:"Nến cần xem lại",collective:"Đoạn cần xem lại",normal:"Không vượt ngưỡng",
    insufficient_data:"Chưa đủ dữ liệu",alert:"Nến cần xem lại",budget_suppressed:"Đã đạt giới hạn cảnh báo",insufficient_reference:"Thiếu mẫu đối chiếu"});
  const sources=Object.freeze({SYNTHETIC:"Dữ liệu giả lập",HISTORICAL_REPLAY:"Dữ liệu lịch sử",
    OFFLINE_RECEIPT_REHEARSAL:"Dữ liệu đã lưu",RECEIVED_MARKET_DATA:"Dữ liệu vừa nhận",NO_SESSION:"Chưa có dữ liệu"});
  function sourceName(mode){return sources[mode]||"Nguồn chưa xác minh";}
  function sourceStatus(mode){
    if(mode==="SYNTHETIC"||mode==="HISTORICAL_REPLAY")return sourceName(mode)+" · Phát lại, không trực tiếp";
    if(mode==="OFFLINE_RECEIPT_REHEARSAL")return "Dữ liệu đã lưu · Chấm lại, không trực tiếp";
    if(mode==="RECEIVED_MARKET_DATA")return "Dữ liệu vừa nhận · Chưa chứng nhận theo dõi trực tiếp";
    return sourceName(mode);
  }
  function modelLabel(id,fallback=id){return modelNames[id]||fallback;}
  function validDate(value){return typeof value==="string"&&Number.isFinite(Date.parse(value));}
  function date(value){return validDate(value)?new Date(value).toLocaleDateString("vi-VN",{timeZone:"Asia/Ho_Chi_Minh",day:"2-digit",month:"2-digit",year:"numeric"}):"Chưa có";}
  function dateTime(value){return validDate(value)?date(value)+" · "+new Date(value).toLocaleTimeString("vi-VN",{timeZone:"Asia/Ho_Chi_Minh",hour:"2-digit",minute:"2-digit",second:"2-digit"}):"Chưa có";}
  function pollSummary(value){
    if(value.source_mode==="OFFLINE_RECEIPT_REHEARSAL")return "";
    const done=value.polls_completed,max=value.max_polls,calls=value.provider_calls;
    const count=v=>Number.isInteger(v)&&v>=0;
    if(count(done)&&count(max)&&max>0)return `${done}/${max} lượt nhận`+(count(calls)?` · ${calls} lần gọi nguồn`:" · số lần gọi nguồn chưa xác minh");
    return count(calls)?`${calls} lần gọi nguồn`:"Chưa có thông tin lượt nhận";
  }
  function monitorState(value){
    const messages={NOT_STARTED:["Chưa có dữ liệu","Chọn Replay để xem dữ liệu đã có."],
      LOADING:["Đang đọc dữ liệu","Đang kiểm tra kết quả của mô hình đã chọn."],
      INSUFFICIENT_HISTORY:["Chưa đủ dữ liệu để chấm","Các nến hiện có chưa tạo được cửa sổ hợp lệ cho mô hình đã chọn."],
      BLOCKED_UNKNOWN_CALENDAR:["Chưa xác minh lịch phiên",`Chưa thể nhận thêm dữ liệu cho ngày ${date(value.calendar_date)}.`],
      OUTSIDE_SESSION:["Ngoài giờ giao dịch","Có thể xem dữ liệu đã lưu trong Replay."],
      BLOCKED_HISTORY_EVIDENCE:["Thiếu lịch sử đã kiểm chứng","Chưa chấm mô hình; dữ liệu tham chiếu cần được kiểm tra."],
      BLOCKED_MEASURED_DELAY:["Chưa xác minh độ trễ nguồn","Cần số đo đúng mã và phiên trước khi nhận dữ liệu mới."],
      BLOCKED_CONTEXT_OR_ARTIFACT:["Không xác minh được mô hình","Tệp mô hình hoặc đầu vào không khớp bản đã lưu."],
      BLOCKED_ARTIFACT:["Không xác minh được mô hình","Tệp mô hình không khớp bản đã lưu."],
      BLOCKED_RUNTIME:["Lượt theo dõi gặp lỗi","Đã dừng xử lý; kết quả trước đó không phải kết quả mới."],
      BLOCKED_PROVIDER_ACCESS:["Nguồn từ chối truy cập","Đã dừng nhận dữ liệu; không tự gọi lại."],
      BLOCKED_PROVIDER_QUOTA:["Nguồn giới hạn lượt gọi","Đã dừng theo ngân sách nhận dữ liệu."],
      BLOCKED_PROVIDER_HTTP:["Nguồn trả về lỗi","Chưa có dữ liệu mới hợp lệ để chấm."],
      RETRY_EXHAUSTED:["Chưa nhận được dữ liệu","Đã hết số lần thử cho phép."],
      BLOCKED_BEFORE_MODEL_FREEZE:["Phiên chưa phù hợp để thử mô hình","Không dùng dữ liệu trước mốc khóa mô hình làm thử nghiệm mới."],
      BLOCKED_GAPS:["Thiếu nến","Tạm dừng chấm; không tự điền dữ liệu thiếu."],
      BLOCKED_READINESS:["Dữ liệu chưa đủ điều kiện","Kiểm tra nến thiếu, độ trễ và lịch phiên trong chi tiết dữ liệu."],
      STALE:["Dữ liệu đã trễ","Chưa chấm kết quả mới."],SNAPSHOT_STALE:["Dữ liệu chưa cập nhật","Đang giữ lần nhận gần nhất; chưa có kết quả mới."],
      FINISHED:["Lượt theo dõi đã kết thúc","Kết quả lần nhận cuối được giữ lại."],STOPPED:["Đã dừng theo dõi","Kết quả đã lưu vẫn có thể xem."],
      VALIDATED_NO_NETWORK:["Đã kiểm tra điều kiện","Chưa khởi chạy nguồn dữ liệu."],
      READY_FOR_SCORE:["Nến sẵn sàng","Chưa có kết quả chấm mô hình."],WAITING_FIRST_CLOSED_BAR:["Đang chờ nến đóng","Chưa có nến đủ điều kiện để chấm."]};
    if(["AI_READY","SCORED_OFFLINE","SCORED_LIVE_CANDIDATE"].includes(value.state)){
      const count=Array.isArray(value.events)?value.events.length:0;
      return ["Đã chấm dữ liệu",count?`${value.symbol||"Mã đang xem"} · phiên ${date(value.calendar_date)} · ${count} kết quả ban đầu`:"Chưa có kết quả ban đầu trong lần đọc này."];
    }
    return messages[value.state]||["Trạng thái chưa xác minh","Chọn Làm mới để kiểm tra lại dữ liệu."];
  }
  function chartHeight(normal){return typeof document!=="undefined"&&document.body.dataset.chartExpanded==="true"?Math.max(480,Math.min(800,Math.round(innerHeight*.65))):normal;}
  function attachLayout(){document.querySelectorAll('[data-chart-expand]').forEach(button=>button.addEventListener('click',()=>{
    const expanded=document.body.dataset.chartExpanded!=="true";document.body.dataset.chartExpanded=String(expanded);
    button.setAttribute('aria-pressed',String(expanded));button.textContent=expanded?"Thu gọn biểu đồ":"Mở rộng biểu đồ";
    window.dispatchEvent(new Event('stock-chart-layout'));
  }));}
  return Object.freeze({modelNames,decisions,sourceName,sourceStatus,modelLabel,date,dateTime,pollSummary,monitorState,chartHeight,attachLayout});
})();
if(typeof module!=="undefined")module.exports=StockMonitorUI;
if(typeof window!=="undefined"){window.StockMonitorUI=StockMonitorUI;document.addEventListener('DOMContentLoaded',StockMonitorUI.attachLayout);}
