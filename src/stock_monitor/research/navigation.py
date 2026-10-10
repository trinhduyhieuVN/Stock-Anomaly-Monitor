"""Read-only P01 journal/data views. No new scoring, source access or model fit."""
import pandas as pd
import streamlit as st

from stock_monitor.upgrade.contracts import utc_time
from stock_monitor.upgrade.ui_service import event_csv
from .presentation import LABELS, local_time


def journal_rows(snapshot, *, bundle_key, symbol, session):
    """Keep only reviews of known versions in this source/symbol/session at T."""
    cutoff = utc_time(snapshot["as_of"])
    versions = {(row["event_id"], row["version"]): row for row in snapshot["versions"]
                if row["symbol"] == symbol and row["session_id"] == session and utc_time(row["available_at"]) <= cutoff}
    latest = {}
    for event_id, version in versions:
        latest[event_id] = max(latest.get(event_id, 0), version)
    output = []
    for review in snapshot["reviews"]:
        event = versions.get((review["event_id"], review["event_version"]))
        if event is None or utc_time(review["available_at"]) > cutoff:
            continue
        extra = {name: snapshot[name] for name in ("availability_basis", "model", "received_at_utc") if name in snapshot}
        output.append({**review, **extra, "source_mode": event["source_mode"], "symbol": symbol, "session_id": session,
                       "event_end": event["event_end"], "bundle_sha256": bundle_key, "export_as_of": snapshot["as_of"],
                       "is_latest_visible_version": latest.get(review["event_id"]) == review["event_version"]})
    return output


def render_journal(snapshot, *, bundle_key, symbol, session, replay):
    st.subheader("Nhật ký rà soát")
    st.write("Nhận xét cho mã/ngày và phạm vi thời gian đang chọn. Bộ lọc sự kiện ở Quan sát không xóa hoặc giấu nhật ký này.")
    rows = journal_rows(snapshot, bundle_key=bundle_key, symbol=symbol, session=session)
    st.session_state.research_journal_csv = event_csv(rows) if rows else b""
    if not rows:
        st.info("Chưa có nhận xét trong phạm vi đang xem. " +
                ("Nhận xét viết sau T không xuất hiện trong replay." if replay else "Mở Quan sát, chọn một sự kiện rồi ghi lý do và bằng chứng."))
        return
    frame = pd.DataFrame([{"Nến (UTC+7)": local_time(row["event_end"]), "Người rà soát": row["reviewer"],
                           "Cách rà soát": "Có AI hỗ trợ" if row["reviewer_kind"] == "ai_assisted" else "Người trực tiếp",
                           "Nhận định": LABELS[row["label"]], "Lý do": row["reason"], "Phiên bản": row["event_version"],
                           "Bản sự kiện": "Hiện tại" if row["is_latest_visible_version"] else "Bản trước",
                           "Giờ lưu thực (UTC+7)": pd.Timestamp(row["available_at"]).tz_convert("Asia/Ho_Chi_Minh").strftime("%Y-%m-%d %H:%M:%S")}
                          for row in rows])
    st.dataframe(frame, hide_index=True, width="stretch", placeholder="—")
    st.download_button("Tải nhật ký đang xem (CSV)", st.session_state.research_journal_csv,
                       file_name=f"{'historical' if snapshot.get('source_mode') == 'HISTORICAL_REPLAY' else 'synthetic'}_reviews_{symbol}_{session}.csv", mime="text/csv")
    st.write("Nhận xét gắn đúng phiên bản, chỉ thêm mới; bản trước không bị sửa. Nhãn có AI hỗ trợ không được coi là nhãn chuyên gia.")


def render_data(bundle, snapshot, *, config_name, journal_path, root):
    st.subheader("Dữ liệu & nghiên cứu")
    st.write("Nguồn đang dùng: fixture tổng hợp có manifest đã kiểm hash. Không phải dữ liệu thị trường hoặc phiên thử LIVE.")
    st.dataframe(pd.DataFrame([
        {"Thông tin": "Đơn vị", "Trạng thái": "Giá/khối lượng giả lập; nến 5 phút"},
        {"Thông tin": "Nến trong phạm vi đang xem", "Trạng thái": str(len(snapshot["bars"]))},
        {"Thông tin": "Tập fixture", "Trạng thái": f"{len(bundle['rows'])} observations, {len(bundle['sessions'])} ngày chọn được"},
        {"Thông tin": "Kiểm toàn vẹn", "Trạng thái": "Manifest và đủ member đã được kiểm trước mỗi lần đọc cache"},
        {"Thông tin": "Phương pháp", "Trạng thái": "Điểm / đoạn suffix 3, 6, 12; calibration giả lập đã ghim"},
        {"Thông tin": "Dự báo", "Trạng thái": "Mốc so sánh lợi suất bằng 0, h1/3; không phải dự đoán giá bằng 0"},
        {"Thông tin": "V1 robust / IF / LSTM", "Trạng thái": "Không nạp ở nguồn giả lập; chọn Lịch sử thật để xem v1, không dùng calibration giả lập chấm nguồn thật"},
        {"Thông tin": "Lịch sử thật / phiên thử", "Trạng thái": "Lịch sử có nguồn riêng; phiên thử chưa mở, cần gate P06"},
        {"Thông tin": "UX thực tế / hiệu quả thị trường", "Trạng thái": "Chưa kiểm chứng bởi các tests giao diện"},
    ]), hide_index=True, width="stretch", placeholder="—")
    st.write("Quyền nguồn: fixture synthetic có thể chia sẻ. Dữ liệu thật local-only vẫn còn sáu xác nhận pending; UI này không thay đổi trạng thái đó.")
    with st.expander("Provenance và nhật ký local"):
        st.write("Cấu hình: " + config_name)
        st.code("Bundle SHA256: " + bundle["key"])
        st.code("Journal: " + journal_path.relative_to(root).as_posix())
        st.write("Nhật ký tách theo source/bundle; không ghi vào journal demo v2 cũ. Không upload model tùy ý hoặc sửa manifest để vượt kiểm hash.")
    with st.expander("Từ điển ngắn"):
        st.write("Điểm: một nến cần xem. Đoạn: một chuỗi nến cần xem. Chưa đủ dữ liệu không có nghĩa bình thường.")
        st.write("Hạng: vị trí trong mẫu tham chiếu, không phải xác suất. Đang chờ: chưa có nến đích; không chấm kết quả: thiếu phần tiếp theo hợp lệ.")
        st.write("Replay chỉ biết đến T. Tin giả lập không chứng minh nguyên nhân. Cảnh báo chỉ ưu tiên rà soát, không phải khuyến nghị giao dịch.")
