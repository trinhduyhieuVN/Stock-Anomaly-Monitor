"""Local-only historical UI. No network client, training or real-v2 calibration."""
from datetime import datetime, timezone
import os
import sqlite3
import uuid

import pandas as pd
import streamlit as st

from stock_monitor.io import DataContractError
from stock_monitor.services import MODELS
from stock_monitor.upgrade.journal import Journal
from stock_monitor.upgrade.ui_service import filtered_events
from .history import load_historical_bundle, verify_history_binding, historical_session, historical_csv
from .history_review import review_snapshot
from .navigation import render_journal
from .presentation import KINDS, LABELS, local_time, number, timeline_figure
from .analysis_ui import render_summary, render_explanation

MANIFEST_SHA = "ab42fa8720162c301a9b4a6affecb767986c62147ab6d649d7c406a1e5811d1e"


@st.cache_data(show_spinner=False, max_entries=2)
def cached_history(root, manifest_name, binding_key):
    # Caller verifies the binding on every rerun; session() also checks every pin.
    return load_historical_bundle(root, expected_manifest_sha256=MANIFEST_SHA, manifest_name=manifest_name)


def table(frame):
    st.dataframe(frame.style.format(precision=4, na_rep="—"), hide_index=True, width="stretch", placeholder="—")


def render_history(root, *, level, area, full_session, help_visibility, reset_filters):
    # Source-specific widget IDs prevent the browser retaining synthetic labels
    # while server-side state has already switched to the historical defaults.
    def full_history_session():
        st.session_state.research_history_mode = "Toàn phiên có sẵn"
    def reset_history_filters():
        st.session_state.research_history_kinds = list(KINDS)
        st.session_state.research_history_review_filter = "all"
    # Never leave a previous source/model download accessible when validation fails.
    st.session_state.research_export_csv = b""
    st.session_state.research_journal_csv = b""
    st.info("LỊCH SỬ THẬT · Nến 5 phút · Offline, không phải LIVE · Chỉ nghiên cứu local, quyền nguồn pending; không phân phối.")
    with st.expander("Lần đầu dùng? Đọc nhanh cách xem", expanded=st.session_state.get("research_help_open", True)):
        st.write("1. Chọn mã, ngày và model ở thanh bên. Mặc định toàn phiên mới nhất để thấy đủ nến.")
        st.write("2. Chọn một nến dưới biểu đồ để đọc OHLCV, điểm số, ngưỡng và giới hạn bằng chứng.")
        st.write("3. Ghi nhận xét ở Rà soát; xem lại ở Nhật ký. Replay chỉ xem đến T và không dùng nhận xét viết sau T.")
        st.button("Ẩn hướng dẫn", on_click=help_visibility, args=(False,))
    try:
        manifest_name = os.environ.get("STOCK_MONITOR_RESEARCH_HISTORY_MANIFEST", "demo/manifest.local.json")
        binding = verify_history_binding(root, manifest_name, MANIFEST_SHA)
        bundle = cached_history(str(root), manifest_name, binding["key"])
        st.sidebar.subheader("Mã & thời gian")
        symbol = st.sidebar.selectbox("Mã cổ phiếu", bundle["symbols"], key="research_history_symbol")
        session = st.sidebar.selectbox("Ngày lịch sử", list(reversed(bundle["sessions"])), key="research_history_session")
        model_name = st.sidebar.selectbox("Model v1 đã ghim", list(MODELS), key="research_history_model")
        model = MODELS[model_name]
        mode = st.sidebar.radio("Phạm vi thời gian", ["Toàn phiên có sẵn", "Replay tại T"], key="research_history_mode")
        context = (bundle["key"], symbol, session, model)
        if st.session_state.get("research_history_context") != context:
            st.session_state.research_history_context = context
            st.session_state.research_history_cursor = 0
            for key in ("research_history_event", "research_history_selected_event", "research_history_reviewer", "research_history_reason", "research_history_reference", "research_history_saved"):
                st.session_state.pop(key, None)
        timeline = bundle["frame"].loc[bundle["frame"].symbol.eq(symbol) & bundle["frame"].session_id.eq(session), "timestamp"].tolist()
        if not timeline:
            raise DataContractError("Không có nến cho mã/ngày này")
        replay = mode == "Replay tại T"
        def move(reset=False):
            st.session_state.research_history_cursor = 0 if reset else min(st.session_state.research_history_cursor + 1, len(timeline) - 1)
        with st.sidebar.expander("Điều khiển Replay", expanded=replay):
            st.select_slider("Đã biết đến T (UTC+7)", options=list(range(len(timeline))), key="research_history_cursor",
                             format_func=lambda i: local_time(timeline[i]), disabled=not replay)
            st.button("Bước tiếp", on_click=move, disabled=not replay or st.session_state.research_history_cursor == len(timeline) - 1,
                      type="primary" if replay else "secondary", width="stretch")
            st.button("Về nến đầu", on_click=move, kwargs={"reset": True}, disabled=not replay, width="stretch")
            st.caption("Chọn Replay tại T để xem từng nến. Toàn phiên giữ đầy đủ nến đã có, không phải LIVE.")
        cutoff = timeline[st.session_state.research_history_cursor] if replay else timeline[-1]
        history = historical_session(bundle, symbol=symbol, session_id=session, model=model, as_of=cutoff.isoformat())
        journal_directory = (root / os.environ.get("STOCK_MONITOR_RESEARCH_JOURNAL", ".local/research_ui")).resolve()
        if not journal_directory.is_relative_to(root):
            raise DataContractError("Nhật ký phải nằm trong project")
        journal_path = journal_directory / bundle["key"] / "historical" / model / "journal.sqlite"
        journal = Journal(journal_path)
        snapshot = review_snapshot(bundle, history, journal, replay=replay)
        st.session_state.research_export_csv = historical_csv(history)
    except (OSError, ValueError, KeyError, TypeError, DataContractError, sqlite3.Error) as exc:
        st.error(f"Dừng đọc lịch sử: artifact hoặc nhật ký không hợp lệ: {exc}")
        st.info("Kiểm tra demo/manifest.local.json và đủ artifact đã ghim; không sửa hash để bỏ kiểm tra. Có thể chọn nguồn Giả lập để tiếp tục demo.")
        return
    with st.sidebar.expander("Bộ lọc nâng cao"):
        kinds = st.multiselect("Loại sự kiện", list(KINDS), default=list(KINDS), format_func=KINDS.get, key="research_history_kinds")
        review_filter = st.selectbox("Nhận xét", ["all", "reviewed", "unreviewed"],
            format_func={"all": "Tất cả", "reviewed": "Đã có nhận xét", "unreviewed": "Chưa có nhận xét"}.get, key="research_history_review_filter")
        st.caption("Bộ lọc chỉ đổi danh sách sự kiện, không xóa nến hay thay ngưỡng.")
    events = filtered_events(snapshot, kinds=kinds, stages=["initial"], review_status=review_filter)
    st.write(f"{symbol} · {session} · {model_name} · Nến đến {local_time(cutoff)} UTC+7.")
    st.caption("Replay theo giả định bar-end, không có giờ nhận lịch sử từng nến. Model được ghim về sau; không khẳng định model đã tồn tại tại T. Giá VND/volume cổ phiếu là giả định profile research chưa được nguồn xác nhận.")
    if "research_history_event" in st.session_state:
        st.session_state.research_history_selected_event = st.session_state.research_history_event
    if area == "Nhật ký rà soát":
        render_journal(snapshot, bundle_key=bundle["key"], symbol=symbol, session=session, replay=replay)
        return
    if area == "Dữ liệu & nghiên cứu":
        st.subheader("Dữ liệu & nghiên cứu")
        st.write(f"{len(bundle['frame'])} nến đã lưu · {len(bundle['sessions'])} phiên · FPT, HPG, VCB. Phạm vi đang xem: {len(history['bars'])} nến.")
        st.write("Score v1: robust z-score / Isolation Forest / LSTM đã lưu; so khớp artifact với rtol=0, atol=0. Không huấn luyện hoặc suy luận mới trong UI.")
        st.write("Quyền nguồn: pending, local-only, không phân phối. Sáu xác nhận nguồn vẫn pending; giá/volume dùng giả định research, lịch observed-only, điều chỉnh giá chưa xác nhận.")
        st.write("Giờ nhận lịch sử: không có. Thời điểm đọc/projection hiện tại (không phải receipt): " + history["processed_at_utc"])
        st.write("V2 trên dữ liệu thật: chưa có calibration được chấp nhận. Tin, forecast và độ trễ LIVE: chưa có bằng chứng cho nguồn này.")
        st.write("Hiệu quả: chưa đo mới; dữ liệu cũ đã được khám phá, không phải holdout độc lập mới. UI chạy đúng không chứng minh phát hiện có hiệu quả.")
        with st.expander("Provenance và nhật ký local"):
            st.code("Manifest SHA256: " + MANIFEST_SHA)
            st.code("Bundle SHA256: " + bundle["key"])
            st.code("Model run: " + history["provenance"]["model_run"])
            st.code("Journal: " + journal_path.relative_to(root).as_posix())
            st.json(history["provenance"]["source_confirmations"])
        return
    bars = history["bars"]
    st.subheader("Nến và các điểm cần xem")
    st.caption(f"{len(bars)} nến · {int(history['scores'].score.notna().sum())} nến có score v1 · {len(events)} sự kiện theo bộ lọc. Nến chưa đủ cửa sổ vẫn được giữ.")
    if replay and len(bars) <= 1:
        st.info("Bạn đang ở nến đầu, không phải lỗi tải. Bước tiếp để thấy thêm nến hoặc Xem toàn phiên.")
        st.button("Xem toàn phiên", on_click=full_history_session, type="primary", width="stretch")
    figure = timeline_figure(bars, events)
    for trace in figure.data:
        if trace.type == "candlestick":
            trace.name = "Nến OHLC lịch sử"
        elif trace.type == "bar":
            trace.name = "Khối lượng lịch sử"
    figure.update_yaxes(title_text="Giá (giả định VND)", row=1, col=1)
    st.plotly_chart(figure, width="stretch", theme=None, config={"displaylogo": False})
    st.caption("Xanh: đóng ≥ mở; đỏ: đóng < mở. Râu nến: cao/thấp. Hình thoi vàng: cửa sổ v1 vượt ngưỡng đã ghim, không phải kết luận gian lận.")
    render_summary(bars, symbol=symbol, session_id=session, as_of=history["as_of"], source_mode="HISTORICAL_REPLAY", level=level)
    st.subheader("Chọn nến để hiểu lý do")
    table(pd.DataFrame([{"Nến (UTC+7)": local_time(row["event_end"]), "Trạng thái": KINDS[row["kind"]],
                         "Score v1 (không phải xác suất)": row["score"], "Ngưỡng đã ghim": row["threshold"]} for row in events]))
    st.download_button("Tải nến và điểm số (CSV)", st.session_state.research_export_csv,
                       file_name=f"historical_{symbol}_{session}_{model}.csv", mime="text/csv")
    st.caption("CSV gồm toàn bộ nến đến T, không bị bộ lọc sự kiện xóa. Dấu — là thiếu hoặc không áp dụng, không phải số 0.")
    if not events:
        st.info("Không có sự kiện theo bộ lọc này. Nến vẫn hiển thị; không giảm ngưỡng để tạo cảnh báo.")
        st.button("Đặt lại bộ lọc", on_click=reset_history_filters)
        return
    lookup = {f"{local_time(row['event_end'])} · {KINDS[row['kind']]} · {row['event_id'][:8]}": row for row in events}
    if st.session_state.get("research_history_event") not in lookup:
        saved = st.session_state.get("research_history_selected_event")
        st.session_state.research_history_event = saved if saved in lookup else next((key for key, row in lookup.items() if row["kind"] == "point"), next(iter(lookup)))
    event = lookup[st.selectbox("Sự kiện cần xem", list(lookup), key="research_history_event")]
    st.session_state.research_history_selected_event = st.session_state.research_history_event
    detail = snapshot["details"][event["event_id"]]
    st.subheader("Điều gì đáng chú ý?")
    st.write(f"Nến {local_time(event['event_end'])}: mở {number(detail['open'])}, cao {number(detail['high'])}, thấp {number(detail['low'])}, đóng {number(detail['close'])}; volume {number(detail['volume'])}.")
    if event["kind"] == "insufficient_data":
        st.write("Chưa đủ cửa sổ hợp lệ cho score v1. Không coi đây là nến bình thường và không lấp bằng tương lai.")
    else:
        st.write(f"{model_name}: score {number(event['score'])}, ngưỡng {number(event['threshold'])}. " +
                 ("Vượt ngưỡng đã ghim: ưu tiên xem lại cửa sổ, chưa biết nguyên nhân." if event["kind"] == "point" else "Không được đánh dấu bởi ngưỡng này; không chứng minh an toàn hoặc bình thường."))
    st.write("So với gì? Score và ngưỡng v1 đã lưu, dùng cửa sổ quá khứ theo feature bundle; không dùng mẫu tham chiếu giả lập, không phải xác suất gian lận.")
    st.write("Còn thiếu? Receipt/revision lịch sử, tin đã biết tại T, nhãn chuyên gia mới và holdout độc lập. Đây là nghiên cứu thăm dò trên dữ liệu cũ.")
    st.write("Bước tiếp theo: kiểm tra cửa sổ và chất lượng dữ liệu, ghi nhận xét kèm bằng chứng; không đặt lệnh theo cảnh báo.")
    render_explanation(bars, event, detail, as_of=history["as_of"], source_mode="HISTORICAL_REPLAY", model=model)
    if level == "Nghiên cứu":
        with st.expander("Cửa sổ và điểm số gốc", expanded=True):
            fields = ["timestamp", "window_start", "split", "quality_status", "feature_eligible", "feature_reasons", "score_status", "score", "threshold", "alert", "reason"]
            table(pd.DataFrame([{name: detail.get(name) for name in fields}]))
            if pd.notna(detail["window_start"]):
                window = bundle["frame"].loc[bundle["frame"].symbol.eq(symbol) & bundle["frame"].timestamp.between(detail["window_start"], detail["timestamp"])]
                table(window[["timestamp", "open", "high", "low", "close", "volume"]])
            if model == "lstm":
                table(pd.DataFrame([{name: detail[name] for name in detail if name.startswith(("error_", "share_"))}]))
        st.caption("Không có đoạn suffix v2, forecast hay tin cho nguồn thật này; không thay missing bằng 0 hoặc dùng calibration synthetic.")
    st.subheader("Rà soát của bạn")
    reviews = [row for row in snapshot["reviews"] if row["event_id"] == event["event_id"]]
    if st.session_state.pop("research_history_saved", False):
        st.success("Đã lưu nhận xét mới theo giờ hiện tại; bản cũ không bị sửa.")
    if reviews:
        table(pd.DataFrame(reviews))
    else:
        st.caption("Chưa có nhận xét cho phiên bản này ở thời điểm đang xem.")
    if replay:
        st.info("Chuyển Toàn phiên có sẵn để viết nhận xét. Replay không hiển thị nhận xét từ tương lai.")
        return
    with st.form("research_history_review_form"):
        reviewer = st.text_input("Người rà soát", key="research_history_reviewer")
        reviewer_kind = st.selectbox("Cách rà soát", ["human", "ai_assisted"], format_func={"human": "Người trực tiếp", "ai_assisted": "Có AI hỗ trợ"}.get)
        label = st.selectbox("Nhận định", list(LABELS), format_func=LABELS.get)
        reason = st.text_area("Lý do và bằng chứng", key="research_history_reason")
        reference = st.text_input("Link tham chiếu (nếu có)", key="research_history_reference")
        submitted = st.form_submit_button("Lưu nhận xét mới", type="primary", width="stretch")
    if submitted:
        stamp = datetime.now(timezone.utc).isoformat()
        review = {"schema_version": 2, "review_id": uuid.uuid4().hex, "event_id": event["event_id"], "event_version": event["version"],
                  "reviewer": reviewer, "reviewer_kind": reviewer_kind, "label": label, "reason": reason,
                  "reviewed_at": stamp, "available_at": stamp, "reference_url": reference}
        try:
            journal.append("reviews", review)
            st.session_state.research_history_saved = True
            st.rerun()
        except (DataContractError, ValueError, sqlite3.Error) as exc:
            st.error(f"Chưa lưu nhận xét: {exc}. Nhập tên và lý do; nhãn AI hỗ trợ không được coi là nhãn chuyên gia.")
