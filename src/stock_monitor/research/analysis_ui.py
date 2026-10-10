"""Native read-only presentation of descriptive statistics, shared by both sources."""
import streamlit as st

from stock_monitor.io import DataContractError
from .analysis import summarize_bars, summary_text, explain_method
from .presentation import local_time


def render_summary(bars, *, symbol, session_id, as_of, source_mode, level):
    st.subheader("Tóm tắt diễn biến đã quan sát")
    st.session_state.research_analysis_summary = None
    try:
        value = summarize_bars(bars, symbol=symbol, session_id=session_id, as_of=as_of, source_mode=source_mode)
    except (DataContractError, ValueError, TypeError) as exc:
        st.warning(f"Chưa tổng hợp được: {exc}. Kiểm tra dữ liệu, không điền số liệu giả.")
        return
    st.session_state.research_analysis_summary = value
    label = "Giả lập" if source_mode == "SYNTHETIC" else "Lịch sử thật, replay theo giả định bar-end"
    st.caption(f"{label} · {symbol} · {session_id} · Chỉ nến đã biết đến {local_time(as_of)} UTC+7. Mô tả, không dự báo hay tín hiệu mua/bán.")
    for line in summary_text(value):
        st.write(line)
    if level == "Nghiên cứu":
        with st.expander("Cách tính tóm tắt"):
            st.write("Tối đa 6 nến liên tiếp: (đóng cuối / mở đầu − 1) × 100%; biên độ = (cao nhất − thấp nhất) / mở đầu × 100%. "
                     "Volume: nến cuối / trung vị tối đa 20 nến trước, tối thiểu 6; không dùng nến cuối trong mẫu. Cùng mã/ngày/đoạn, không fit hay đổi ngưỡng.")
            st.json(value)


def render_explanation(bars, event, detail, *, as_of, source_mode, model=None):
    st.subheader("Vì sao cần xem?")
    st.session_state.research_analysis_event = None
    # Selecting an old event in full-session mode must not describe later bars.
    from stock_monitor.upgrade.contracts import utc_time
    cutoff = utc_time(str(as_of))
    # Synthetic receipt can lag bar-end. Keep the already materialized revision
    # known at T, but exclude every bar ending after the selected event.
    prefix = bars.loc[bars.timestamp.le(utc_time(event["event_end"]))]
    try:
        value = summarize_bars(prefix, symbol=event["symbol"], session_id=event["session_id"], as_of=cutoff.isoformat(), source_mode=source_mode)
        st.session_state.research_analysis_event = value
        st.write(f"Bối cảnh nến {local_time(event['event_end'])} UTC+7 theo phiên bản đã biết tại T — không dùng nến sau nến đang chọn; không khẳng định đây là phiên bản lúc phát hiện ban đầu.")
        for line in summary_text(value):
            st.write(line)
    except (DataContractError, ValueError, TypeError) as exc:
        st.warning(f"Bối cảnh chưa đủ tin cậy: {exc}")
    for line in explain_method(event, detail, model=model):
        st.write(line)
    st.caption("Phân biệt: mô tả số liệu → lý do phương pháp đánh dấu → nguyên nhân thị trường. Hai phần đầu không chứng minh phần cuối. Chưa có dự báo giá thật trong bản này.")
