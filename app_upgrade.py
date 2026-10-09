"""Independent synthetic bar/segment review demo; original app.py unchanged."""
from datetime import datetime, timezone
import os
import sqlite3
from pathlib import Path
import uuid

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from stock_monitor.io import DataContractError
from stock_monitor.upgrade.contracts import utc_time
from stock_monitor.upgrade.journal import Journal
from stock_monitor.upgrade.ui_service import load_ui_bundle, replay_session, filtered_events, event_csv, forecast_view

ROOT = Path(__file__).resolve().parent


def local_time(value):
    return pd.Timestamp(value).tz_convert("Asia/Ho_Chi_Minh").strftime("%H:%M:%S")


def show_table(frame, *, column_order=None):
    # Display only: keep missing numeric values in the data/CSV, never fill with 0.
    st.dataframe(frame.style.format(precision=4, na_rep="—"), column_order=column_order, hide_index=True, width="stretch", placeholder="—")


def rank_text(value):
    return "chưa đủ dữ liệu" if value is None or pd.isna(value) else f"{value:.4f}"


def show_full_session():
    st.session_state.v2_mode = "Rà soát hiện tại"


@st.cache_data(show_spinner=False, max_entries=128)
def visible_snapshot(bundle_key, symbol, session_id, logical_cutoff, journal_path):
    # Bundle is hash-verified on every rerun before cache access. Current reviews
    # are queried separately, so caching historical traces cannot hide new input.
    return replay_session(bundle, Journal(journal_path), symbol=symbol, session_id=session_id, as_of=logical_cutoff)


st.set_page_config(page_title="Rà soát điểm & đoạn — demo v2", layout="wide")
st.title("Rà soát điểm & đoạn")
st.caption("SYNTHETIC · Nến 5 phút · Replay logic, không phải LIVE · Chỉ ưu tiên rà soát, không kết luận gian lận hay khuyến nghị giao dịch.")
try:
    config_name = os.environ.get("STOCK_MONITOR_UPGRADE_CONFIG", "configs/upgrade_ui.json")
    bundle = load_ui_bundle(ROOT, config_name)
    journal_name = os.environ.get("STOCK_MONITOR_UPGRADE_JOURNAL", ".local/upgrade_ui/journal.sqlite")
    journal_path = (ROOT / journal_name).resolve()
    if not journal_path.is_relative_to(ROOT):
        raise DataContractError("Nhật ký phải ở trong thư mục project")
    journal = Journal(journal_path)
except (OSError, ValueError, KeyError, TypeError, DataContractError, sqlite3.Error) as exc:
    st.error(f"Không tải được dữ liệu/model/nhật ký synthetic đã ghim: {exc}")
    st.info("Kiểm configs/upgrade_ui.json và đủ bốn member trong demo/upgrade_ui; không upload model hoặc dữ liệu thật để bỏ qua kiểm hash.")
    st.stop()

symbol = st.sidebar.selectbox("Mã cổ phiếu", ["FPT", "HPG", "VCB"], key="v2_symbol")
session_id = st.sidebar.selectbox("Phiên tổng hợp", bundle["sessions"], key="v2_session")
mode = st.sidebar.radio("Chế độ xem", ["Replay tại T", "Rà soát hiện tại"], index=1, key="v2_mode")
st.sidebar.caption("Rà soát hiện tại: xem toàn phiên có sẵn. Replay tại T: chỉ xem dữ liệu đã đến trước T.")
timeline = sorted({row["available_at"] for row in bundle["rows"] if row["symbol"] == symbol and row["session_id"] == session_id}, key=utc_time)
context_key = (bundle["key"], symbol, session_id)
if st.session_state.get("v2_replay_context") != context_key:
    st.session_state.v2_replay_context = context_key
    st.session_state.v2_cursor = 0


def move(reset=False):
    st.session_state.v2_cursor = 0 if reset else min(st.session_state.v2_cursor + 1, len(timeline) - 1)


st.sidebar.select_slider("Thời điểm T", options=list(range(len(timeline))), key="v2_cursor", format_func=lambda i: local_time(timeline[i]), disabled=mode != "Replay tại T")
st.sidebar.button("Bước tiếp", on_click=move, disabled=mode != "Replay tại T" or st.session_state.v2_cursor == len(timeline) - 1)
st.sidebar.button("Reset replay", on_click=move, kwargs={"reset": True}, disabled=mode != "Replay tại T")
cutoff = timeline[st.session_state.v2_cursor] if mode == "Replay tại T" else datetime.now(timezone.utc).isoformat()
if mode == "Replay tại T":
    st.info(f"{symbol} · {session_id} · Đã biết đến T = {local_time(cutoff)} (UTC+7). Reset giữ journal, chỉ đổi phạm vi đã biết.")
else:
    st.info("Rà soát hiện tại trên toàn phiên tổng hợp. Annotation ghi giờ thật hiện tại; không backdate vào replay. Đây không phải feed trực tiếp.")

try:
    logical_cutoff = cutoff if mode == "Replay tại T" else timeline[-1]
    snapshot = visible_snapshot(bundle["key"], symbol, session_id, logical_cutoff, str(journal_path))
    if mode != "Replay tại T":
        event_ids = {row["event_id"] for row in snapshot["events"]}
        snapshot["as_of"] = cutoff
        snapshot["reviews"] = [row for row in journal.query("reviews", as_of=cutoff, symbol=symbol) if row["event_id"] in event_ids]
except (OSError, ValueError, KeyError, DataContractError, sqlite3.Error) as exc:
    st.error(f"Replay bị chặn để giữ hợp đồng dữ liệu: {exc}")
    st.stop()

kinds = st.sidebar.multiselect("Loại sự kiện", ["point", "collective", "normal", "insufficient_data"], default=["point", "collective", "normal", "insufficient_data"], key="v2_kinds")
stages = st.sidebar.multiselect("Giai đoạn", ["initial", "post_event"], default=["initial", "post_event"], key="v2_stages")
review_status = st.sidebar.selectbox("Trạng thái rà soát tại T", ["all", "reviewed", "unreviewed"], key="v2_review_filter")
events = filtered_events(snapshot, kinds=kinds, stages=stages, review_status=review_status)
st.session_state.v2_export_csv = event_csv(events)
bars = snapshot["bars"]
st.subheader("Timeline đã quan sát")
st.caption(f"{len(bars)} nến visible · {len(events)} sự kiện theo bộ lọc · Mỗi endpoint một emission; đoạn chồng lấn chưa được gọi là nhiều TP. Giá/volume là đơn vị giả lập.")
if mode == "Replay tại T" and len(bars) <= 1:
    st.info("Replay đang ở nến đầu tiên, không phải lỗi tải dữ liệu. Bước tiếp hoặc kéo Thời điểm T để xem thêm; chọn Xem toàn phiên để xem ngay biểu đồ đầy đủ.")
    st.button("Xem toàn phiên", on_click=show_full_session)
figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.06, row_heights=[.75, .25])
if not bars.empty:
    times = bars.timestamp.dt.tz_convert("Asia/Ho_Chi_Minh")
    for index, (_, segment) in enumerate(bars.groupby("segment_id", sort=False)):
        figure.add_trace(go.Candlestick(x=segment.timestamp.dt.tz_convert("Asia/Ho_Chi_Minh"),
            open=segment.open, high=segment.high, low=segment.low, close=segment.close,
            name="Nến OHLC synthetic", legendgroup="price", showlegend=index == 0,
            increasing_line_color="#20a486", decreasing_line_color="#e45756"), row=1, col=1)
    figure.add_trace(go.Bar(x=times, y=bars.volume, name="Volume synthetic", marker_color="#5b8ff9"), row=2, col=1)
    for event in events:
        if event["kind"] == "collective":
            figure.add_vrect(x0=pd.Timestamp(event["event_start"]).tz_convert("Asia/Ho_Chi_Minh"), x1=pd.Timestamp(event["event_end"]).tz_convert("Asia/Ho_Chi_Minh"),
                             fillcolor="#8054ad", opacity=.10, line_width=0, row=1, col=1)
        elif event["kind"] == "point":
            match = bars.loc[bars.timestamp.eq(pd.Timestamp(event["event_end"]))]
            if not match.empty:
                figure.add_trace(go.Scatter(x=match.timestamp.dt.tz_convert("Asia/Ho_Chi_Minh"), y=match.close, mode="markers",
                    name="Điểm cần rà soát", marker={"color": "#d62728", "size": 10, "symbol": "diamond"}, showlegend=False), row=1, col=1)
figure.update_layout(height=480, xaxis_rangeslider_visible=False, margin={"l": 15, "r": 15, "t": 10, "b": 20}, hovermode="x unified", legend={"orientation": "h"})
figure.update_yaxes(title_text="Giá giả lập", row=1, col=1)
figure.update_yaxes(title_text="Volume", row=2, col=1)
figure.update_xaxes(tickformat="%H:%M")
if not bars.empty and len(bars) == 1:
    clock = bars.timestamp.iloc[0].tz_convert("Asia/Ho_Chi_Minh")
    figure.update_xaxes(range=[clock - pd.Timedelta(minutes=5), clock + pd.Timedelta(minutes=5)])
st.plotly_chart(figure, width="stretch")
st.caption("Nến xanh: đóng ≥ mở; nến đỏ: đóng < mở. Râu nến thể hiện giá cao/thấp; rê chuột để xem OHLC. Trục thời gian UTC+7; khoảng nghỉ/gap được giữ nguyên.")
st.caption("Vùng tím: suffix collective đã thấy, ranh giới ước lượng tại T; không phải đoạn cuối cùng trong tương lai. Marker hình thoi: point.")

st.subheader("Sự kiện và bằng chứng")
columns = ["kind", "score", "event_end", "stage", "version", "event_start", "available_at", "event_id"]
show_table(pd.DataFrame(events, columns=columns))
st.download_button("Xuất CSV đúng bộ lọc đến T", st.session_state.v2_export_csv, file_name=f"synthetic_{symbol}_{session_id}.csv", mime="text/csv")
st.caption("Chỉ initial: bản quyết định ban đầu version1. Cả hai giai đoạn: version mới nhất visible. Rank không phải xác suất; insufficient_data không được coi là normal.")
st.caption("Dấu —: chưa có giá trị hoặc không áp dụng; xem trạng thái/lý do bên dưới. CSV giữ giá trị thiếu gốc, không thay bằng số 0.")
if not events:
    st.info("Bộ lọc chưa có sự kiện visible. Chọn thêm loại/giai đoạn, đổi trạng thái review hoặc bước tiếp T; không giảm ngưỡng để tạo cảnh báo.")
    st.stop()

lookup = {f"{local_time(row['event_end'])} · {row['kind']} · v{row['version']} · {row['event_id'][:8]}": row for row in events}
default_event = next((i for i, row in enumerate(events) if row["kind"] in {"point", "collective"}), 0)
label = st.selectbox("Sự kiện cần xem", list(lookup), index=default_event, key="v2_event")
event = lookup[label]
detail = snapshot["details"][event["event_id"]]
st.write("Lý do: " + ", ".join(event["reason_codes"]))
evidence_tab, forecast_tab, history_tab, review_tab = st.tabs(["Bối cảnh & candidates", "Forecast & outcome", "Versions & tin đã biết", "Rà soát thủ công"])
with evidence_tab:
    st.write(f"Point rank: {rank_text(detail['point_rank'])} · Joint suffix rank: {rank_text(detail['joint_rank'])} · Độ dài đoạn đã xét: {detail['coverage_mask'] or 'chưa đủ nến'}")
    if event["kind"] == "insufficient_data":
        st.info("Chưa đủ lịch sử trong đoạn hoặc mẫu tham chiếu để tính điểm. Đây không phải kết luận bình thường. Chọn sự kiện muộn hơn hoặc bước tiếp replay; không dùng dữ liệu tương lai để lấp chỗ thiếu.")
    st.write(f"References theo mã: point {detail['point_reference_count']} / suffix-mask {detail['joint_reference_count']} (minimum 100)")
    st.caption("Suffix rank theo symbol/mask, không phải contextual probability. Context dưới đây là proxy giá/volume lagged của U03; real v1 robust/IF/LSTM không có trong bản synthetic độc lập.")
    context = snapshot["context"].loc[snapshot["context"].timestamp.eq(pd.Timestamp(event["event_end"]))]
    fields = ["context_end", "past_trend", "lagged_volatility", "lagged_relative_volume", "context_eligible", "context_reason", "volume_history_count"]
    show_table(context[fields])
    if not context.empty and not context.context_eligible.all():
        st.caption("Bối cảnh chưa đủ lịch sử quá khứ; context_reason ghi nguyên nhân. Các đặc trưng còn thiếu giữ dấu —, không phải giá hoặc volume bằng 0.")
    before = bars.loc[bars.timestamp.le(pd.Timestamp(event["event_end"])) & bars.segment_id.eq(event["segment_id"])].tail(7)
    st.write("Cửa sổ trước endpoint, raw revision đã visible tại T")
    show_table(before[["timestamp", "open", "high", "low", "close", "volume", "bar_revision"]])
    # Point and suffix candidates have different schemas. Do not manufacture a
    # sparse union table that makes point-only fields look like broken data.
    point_candidates = [row for row in detail["candidates"] if "length" not in row]
    suffix_candidates = [{key: value for key, value in row.items() if key != "record"}
                         for row in detail["candidates"] if "length" in row]
    if point_candidates:
        st.write("Ứng viên điểm")
        show_table(pd.DataFrame([{"raw_score": row["raw_score"], "admissible": row["admissible"]} for row in point_candidates]))
    if suffix_candidates:
        st.write("Ứng viên đoạn (suffix)")
        show_table(pd.DataFrame(suffix_candidates))
    if not detail["candidates"]:
        st.info("Chưa có ứng viên: cần thêm nến trong cùng đoạn và lịch sử tính thang đo.")
with forecast_tab:
    forecast_rows = forecast_view(snapshot, event, bundle)
    forecast_statuses = {row["status"] for row in forecast_rows}
    if "pending" in forecast_statuses:
        st.info("Đang chờ nến đích (pending): chưa thể có lợi suất thực tế, residual hay rank. Bước tiếp replay để quan sát kết quả khi dữ liệu đã đến.")
    if any(status.startswith("censored") for status in forecast_statuses):
        st.info("Không chấm kết quả (censored): nến đích vượt ranh giới phiên/đoạn hoặc thiếu dữ liệu hợp lệ. Các ô — là không có kết quả, không phải residual bằng 0.")
    if "mature" in forecast_statuses:
        st.caption("Đã nhận nến đích (mature): lợi suất thực tế và residual đã được tính. Rank chỉ có khi đủ mẫu tham chiếu.")
    show_table(pd.DataFrame(forecast_rows), column_order=["status", "horizon_bars", "prediction", "realized_return", "residual", "rank", "reference_count", "created_at", "target_bar_end", "forecast_id"])
    if not forecast_rows:
        st.warning("Origin nhận quá trễ để forecast trước target; không backfill prediction.")
    st.caption("Zero-return h1/3: prediction = 0 là lợi suất dự đoán bằng 0, không phải giá cổ phiếu bằng 0. pending/censored không có residual0. Mature residual bổ sung bằng chứng, không đổi kind/score ban đầu. Giờ nhận là logic fixture; chưa có live latency.")
with history_tab:
    versions = [row for row in snapshot["versions"] if row["event_id"] == event["event_id"]]
    show_table(pd.DataFrame(versions, columns=columns + ["reason_codes"]))
    with st.expander("Bản ghi kỹ thuật đầy đủ (giữ null gốc để kiểm toán)"):
        st.json(versions)
    if snapshot["news"]:
        st.dataframe(pd.DataFrame(snapshot["news"]), hide_index=True, width="stretch")
    else:
        st.write("Chưa có tin đã biết tại T. Tin synthetic không chứng minh quan hệ nhân quả.")
with review_tab:
    reviews = [row for row in snapshot["reviews"] if (row["event_id"], row["event_version"]) == (event["event_id"], event["version"])]
    if reviews:
        show_table(pd.DataFrame(reviews))
    else:
        st.caption("Chưa có rà soát cho phiên bản sự kiện này tại thời điểm đang xem.")
    if mode == "Replay tại T":
        st.info("Để ghi review, chuyển sang Rà soát hiện tại. Replay không hiển thị annotation được tạo trong tương lai.")
    else:
        with st.form("v2_review_form"):
            reviewer = st.text_input("Người rà soát", key="v2_reviewer")
            reviewer_kind = st.selectbox("Loại reviewer", ["human", "ai_assisted"], key="v2_reviewer_kind")
            reviewed_label = st.selectbox("Nhãn rà soát", ["uncertain", "noteworthy", "ordinary", "data_issue"], key="v2_review_label")
            reason = st.text_area("Lý do và bằng chứng", key="v2_reason")
            reference = st.text_input("Link tham chiếu (nếu có)", key="v2_reference")
            submitted = st.form_submit_button("Lưu review mới")
        if submitted:
            stamp = datetime.now(timezone.utc).isoformat()
            review = {"schema_version": 2, "review_id": uuid.uuid4().hex, "event_id": event["event_id"], "event_version": event["version"],
                      "reviewer": reviewer, "reviewer_kind": reviewer_kind, "label": reviewed_label, "reason": reason,
                      "reviewed_at": stamp, "available_at": stamp, "reference_url": reference}
            try:
                journal.append("reviews", review)
                st.success("Đã lưu review mới theo giờ hiện tại; bản cũ không bị sửa. Refresh để áp bộ lọc reviewed.")
            except (DataContractError, ValueError, sqlite3.Error) as exc:
                st.error(f"Review chưa lưu: {exc}. Nhập tên/lý do không rỗng; AI-assisted không tự nâng thành expert.")
