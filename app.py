"""Vietnamese dashboard for frozen baseline and LSTM research artifacts."""

import json
import os
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stock_monitor.dashboard import latest_baseline_bundle
from stock_monitor.io import DataContractError
from stock_monitor.replay import advance_cursor, replay_dates, replay_prefix, replay_segments
from stock_monitor.services import (MODELS, daily_overview, dashboard_artifact_key, filtered_csv,
                                    filtered_scores, frozen_model_resources, latest_scoring_bundle,
                                    load_monitor_data, load_operator_snapshot, observation_window,
                                    operator_status_view)

ROOT = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def load_data(path, scoring_path, artifact_key):
    return load_monitor_data(path, scoring_path)


@st.cache_resource(show_spinner=False)
def load_models(path, scoring_path, artifact_key):
    return frozen_model_resources(path, scoring_path)


st.set_page_config(page_title="Giám sát bất thường cổ phiếu", layout="wide")
st.title("Giám sát bất thường cổ phiếu 5 phút")
st.caption("Bản nghiên cứu local — score và phân vị ngưỡng không phải xác suất bất thường hay khuyến nghị giao dịch.")

operator_file = os.environ.get("STOCK_MONITOR_OPERATOR_STATUS", "reports/prompt13_operator_status.json")
try:
    operator = operator_status_view(load_operator_snapshot(ROOT, operator_file))
except DataContractError as exc:
    operator = operator_status_view({
        "state": "BLOCKED_DATA_CONTRACT", "live_evidence": False,
        "source_rights_status": "pending", "gap_count": 0, "stale": False,
        "operator_message": f"Không đọc được snapshot operator: {exc}",
    })
with st.expander("Trạng thái cập nhật nến 5 phút", expanded=True):
    getattr(st, operator["severity"])(operator["state_label"])
    st.write(f"**{operator['mode_label']}**")
    st.write(f"Nến đóng mới nhất: **{operator['latest_closed_bar']}** · Quyền nguồn: **{operator['source_rights_status']}**")
    st.write(f"Gap: **{operator['gap_count']}** · Stale: **{'có' if operator['stale'] else 'không'}**")
    st.write(f"Receive time live: **{operator['receive_time']}** · Score duration live: **{operator['score_duration']}**")
    st.caption("Audit timing offline: " + operator["offline_audit"])
    if operator["message"]:
        st.caption(operator["message"])

try:
    configured = os.environ.get("STOCK_MONITOR_BASELINE_DIR")
    bundle_path = Path(configured).resolve() if configured else latest_baseline_bundle(ROOT)
    baseline_metadata_path = bundle_path / "metadata.json"
    baseline_metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
    configured_scoring = os.environ.get("STOCK_MONITOR_SCORING_DIR")
    scoring_path = Path(configured_scoring).resolve() if configured_scoring else latest_scoring_bundle(
        ROOT, baseline_metadata["input_feature_dir"])
    scoring_text = str(scoring_path) if scoring_path else None
    artifact_key = dashboard_artifact_key(bundle_path, scoring_path)
    frame, metadata, features = load_data(str(bundle_path), scoring_text, artifact_key)
    model_resources = load_models(str(bundle_path), scoring_text, artifact_key)
except (OSError, ValueError, KeyError, TypeError, DataContractError) as exc:
    st.error(f"Không thể tải dữ liệu/model: {exc}")
    st.info("Kiểm tra bundle và chạy CLI features, baselines, train-lstm, score-lstm theo README trước khi mở app.")
    st.stop()

symbols = sorted(frame.symbol.unique())
symbol = st.sidebar.selectbox("Mã cổ phiếu", symbols)
replaying = st.sidebar.radio("Chế độ", ["Lịch sử", "Replay test"]) == "Replay test"
test_dates = replay_dates(metadata["feature"])
dates = sorted(frame.loc[frame.symbol.eq(symbol), "session_id"].unique(), reverse=True)
if replaying:
    dates = [date for date in dates if date in test_dates]
if not dates:
    st.warning("Chưa có phiên test để phát lại.")
    st.stop()
session_id = st.sidebar.selectbox("Ngày giao dịch", dates)
available_models = [label for label, name in MODELS.items() if name != "lstm" or metadata["lstm"] is not None]
model_label = st.sidebar.selectbox("Model", available_models)
model = MODELS[model_label]
if metadata["lstm"] is None:
    st.warning("Chưa có LSTM scores/model cho dataset này. Chạy train-lstm rồi score-lstm với cùng feature bundle.")

visible_frame = frame
if replaying:
    timeline = frame.loc[frame.symbol.eq(symbol) & frame.session_id.eq(session_id), "timestamp"].sort_values().tolist()
    context = (artifact_key, symbol, session_id, model)
    if st.session_state.get("replay_context") != context:
        st.session_state.replay_context = context
        st.session_state.replay_cursor = 0

    def move_replay(reset=False):
        st.session_state.replay_cursor = advance_cursor(st.session_state.replay_cursor, len(timeline), reset=reset)

    st.sidebar.select_slider("Thời điểm T", options=list(range(len(timeline))), key="replay_cursor",
                             format_func=lambda position: timeline[position].tz_convert("Asia/Ho_Chi_Minh").strftime("%H:%M"))
    st.sidebar.button("Bước tiếp", on_click=move_replay, disabled=st.session_state.replay_cursor == len(timeline) - 1)
    st.sidebar.button("Reset replay", on_click=move_replay, kwargs={"reset": True})
    cutoff = timeline[st.session_state.replay_cursor]
    visible_frame = replay_prefix(frame, session_id, cutoff, test_dates)
    st.info("Phát lại lịch sử — không phải feed trực tiếp. T = " + cutoff.tz_convert("Asia/Ho_Chi_Minh").isoformat() +
            ". Chỉ hiển thị thông tin đã có đến T; model/scaler/ngưỡng giữ nguyên.")

try:
    selected = filtered_scores(visible_frame, symbol, session_id, model)
except DataContractError as exc:
    st.warning(str(exc))
    st.stop()

local_time = selected.timestamp.dt.tz_convert("Asia/Ho_Chi_Minh")
alerts = selected.loc[selected.alert]
thresholds = selected.threshold.dropna().unique()
threshold_text = f"{thresholds[0]:.4f}" if len(thresholds) else "Chưa có"
metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Số nến", len(selected))
metric_b.metric("Cảnh báo", int(selected.alert.sum()))
metric_c.metric("Ngưỡng validation 98%", threshold_text)

overview_tab, detail_tab, info_tab = st.tabs(["Tổng quan", "Chi tiết", "Thông tin"])
with overview_tab:
    st.subheader(f"Các mã trong ngày {session_id}")
    st.dataframe(daily_overview(visible_frame, session_id, model), hide_index=True, width="stretch")
    st.caption("Nến mới nhất là thời điểm của dữ liệu lịch sử đang xem.")

figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                       row_heights=[0.55, 0.2, 0.25])
figure.add_trace(go.Candlestick(x=local_time, open=selected.open, high=selected.high,
                                low=selected.low, close=selected.close, name="Giá"), row=1, col=1)
figure.add_trace(go.Bar(x=local_time, y=selected.volume, name="Khối lượng", marker_color="#5b8ff9"), row=2, col=1)
if not alerts.empty:
    alert_time = alerts.timestamp.dt.tz_convert("Asia/Ho_Chi_Minh")
    figure.add_trace(go.Scatter(x=alert_time, y=alerts.high, mode="markers", name="Cảnh báo",
                                marker={"color": "#d62728", "size": 11, "symbol": "triangle-down"},
                                customdata=alerts.score, hovertemplate="Score %{customdata:.4f}<extra></extra>"), row=1, col=1)
figure.add_trace(go.Scatter(x=local_time, y=selected.score, mode="lines", name="Score"), row=3, col=1)
figure.add_trace(go.Scatter(x=local_time, y=selected.threshold, mode="lines", name="Ngưỡng",
                            line={"dash": "dash", "color": "#d62728"}), row=3, col=1)
figure.update_layout(height=760, xaxis_rangeslider_visible=False, hovermode="x unified",
                     legend={"orientation": "h", "y": 1.04}, margin={"l": 20, "r": 20, "t": 35, "b": 20})
figure.update_yaxes(title_text="Giá (VND)", row=1, col=1)
figure.update_yaxes(title_text="Khối lượng", row=2, col=1)
figure.update_yaxes(title_text="Score", row=3, col=1)
run_metadata = metadata["lstm"] if model == "lstm" else metadata["baseline"]
with detail_tab:
    st.plotly_chart(figure, width="stretch")
    if replaying:
        segments = replay_segments(selected).rename(columns={"end_timestamp": "Nến cảnh báo cuối đã thấy",
                                                            "ongoing": "Đang mở"})
        st.subheader("Đoạn cảnh báo đã quan sát")
        st.dataframe(segments, hide_index=True, width="stretch")
        st.caption("Đang mở: chưa thấy nến đóng đoạn; không biết ranh giới kết thúc trong tương lai.")
    candidates = alerts if not alerts.empty else selected.loc[selected.score.notna()]
    if candidates.empty:
        st.info("Các nến trong bộ lọc này chưa đủ điều kiện/cửa sổ để chấm điểm.")
    else:
        point_times = candidates.timestamp.tolist()
        point = st.selectbox("Điểm cần xem", point_times,
                             format_func=lambda stamp: stamp.tz_convert("Asia/Ho_Chi_Minh").strftime("%H:%M %d/%m/%Y"))
        bar = candidates.loc[candidates.timestamp.eq(point)].iloc[0]
        st.write(f"Score **{bar.score:.6f}** · Ngưỡng **{bar.threshold:.6f}** · "
                 f"Trạng thái: **{'cảnh báo' if bar.alert else 'dưới ngưỡng'}** · Chất lượng: **{bar.quality_status}**")
        st.caption("Lý do: score vượt phân vị 0.98 validation của mã này." if bar.alert else "Score chưa vượt ngưỡng validation.")
        window = observation_window(features, bar)
        st.subheader("Cửa sổ quan sát 6 nến")
        st.dataframe(window[["timestamp", "open", "high", "low", "close", "volume"] +
                            metadata["feature"]["feature_order"]], hide_index=True, width="stretch")
        if model == "lstm":
            st.subheader("Đóng góp vào lỗi tái tạo tại nến cuối")
            st.dataframe([{"Feature": name, "Lỗi bình phương": bar[f"error_{name}"],
                           "Tỷ trọng lỗi": bar[f"share_{name}"]} for name in metadata["feature"]["feature_order"]],
                          hide_index=True, width="stretch")
            st.caption("Tỷ trọng là phần đóng góp vào lỗi của model; không giải thích nguyên nhân thị trường. Tổng lỗi bằng 0 thì mọi tỷ trọng bằng 0.")
        else:
            st.caption("Baseline chưa cung cấp tỷ trọng lỗi tái tạo; bảng cửa sổ hiển thị các feature quan sát được.")
    st.download_button("Xuất CSV theo bộ lọc", filtered_csv(selected, model, run_metadata["run_id"],
                        metadata["feature"]["feature_order"]),
                       file_name=f"{symbol}_{session_id}_{model}.csv", mime="text/csv")

with info_tab:
    st.subheader("Nguồn và artifact")
    st.write(f"Nguồn: VCI · Interval: 5m · Model: **{model_label}** · Run: `{run_metadata['run_id']}`")
    st.write("Split hiển thị: " + (", ".join(sorted(selected.split.dropna().unique())) or "warm-up/chưa score"))
    st.write("Split theo ngày chung: train 60%, validation 20%, test 20%; scaler/model fit train, threshold lấy validation.")
    pending = [key for key, value in run_metadata["source_confirmations"].items() if value["status"] != "confirmed"]
    st.warning("Nguồn đang ở profile nghiên cứu local; các xác nhận còn pending: " + ", ".join(pending))
    profile = metadata["feature"]["source_effective_profile"]
    st.write(profile.get("limitations", metadata["feature"].get("note", "")))
    st.info("Precision/recall/F1 dữ liệu thật hiện dựa trên 486 nhãn rà soát AI-assisted, không phải ground truth chuyên gia. Precision cao trên 1–2 cảnh báo baseline đi cùng recall rất thấp; loss và số cảnh báo không tự chứng minh chất lượng phát hiện.")
    st.caption("Dữ liệu là lịch sử nghiên cứu; chưa có bằng chứng feed trực tiếp. Ứng dụng đọc artifact đóng băng và cache theo run/checksum; đổi bộ lọc không huấn luyện lại model.")
