"""Display-only causal descriptions; never change scores or infer market causes."""
import math

import numpy as np
import pandas as pd

from stock_monitor.io import DataContractError
from stock_monitor.upgrade.contracts import utc_time

FEATURES = {"log_return": "lợi suất", "range_ratio": "biên độ nến", "body_ratio": "thân nến",
            "volume_surprise": "khối lượng so với lịch sử"}


def summarize_bars(bars, *, symbol, session_id, as_of, source_mode):
    """Describe <=6 latest contiguous closed bars, with <=20 strictly prior volumes.

    Historical availability is a declared bar-end assumption. Synthetic callers
    must supply the verified as-of observation view, including available_at.
    """
    if source_mode not in {"SYNTHETIC", "HISTORICAL_REPLAY"}:
        raise DataContractError("Descriptions require an explicit offline source")
    cutoff = pd.Timestamp(utc_time(str(as_of), "analysis as_of"))
    required = {"symbol", "session_id", "timestamp", "segment_id", "open", "high", "low", "close", "volume"}
    if required - set(bars.columns):
        raise DataContractError("Missing descriptive bar fields")
    frame = bars.loc[bars.symbol.eq(symbol) & bars.session_id.eq(session_id)].copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp.map(lambda v: utc_time(str(v))), utc=True)
    frame = frame.loc[frame.timestamp.le(cutoff)]
    if source_mode == "SYNTHETIC":
        if "available_at" not in frame or "source" not in frame or not frame.source.eq("SYNTHETIC").all():
            raise DataContractError("Synthetic descriptions require observation availability/source")
        available = pd.to_datetime(frame.available_at.map(lambda v: utc_time(str(v))), utc=True)
        if available.lt(frame.timestamp).any():
            raise DataContractError("Availability precedes bar close")
        frame = frame.loc[available.le(cutoff)]
    elif "source" in frame and not frame.source.eq("VCI").all():
        raise DataContractError("Mixed historical source")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    result = {"source_mode": source_mode, "symbol": symbol, "session_id": session_id, "as_of": cutoff.isoformat(),
              "status": "unavailable", "reason": "no_visible_bars", "window_count": 0,
              "price_change_pct": None, "range_pct": None, "current_body_pct": None,
              "volume_ratio": None, "volume_reference_count": 0, "volume_reference_median": None,
              "volume_reason": "insufficient_prior_bars", "window_start": None, "window_end": None,
              "volume_reference_start": None, "volume_reference_end": None, "boundary_limited": False}
    if frame.empty:
        return result
    if frame.timestamp.duplicated().any():
        raise DataContractError("Duplicate revisions: pass the verified materialized view")
    if not frame.timestamp.dt.tz_convert("Asia/Ho_Chi_Minh").dt.strftime("%Y-%m-%d").eq(session_id).all():
        raise DataContractError("Bar dates differ from selected session")
    values = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(np.isfinite(values.to_numpy()).all(axis=1), index=frame.index)
    valid &= values[["open", "high", "low", "close"]].gt(0).all(axis=1) & values.volume.ge(0)
    valid &= values.high.ge(values[["open", "low", "close"]].max(axis=1))
    valid &= values.low.le(values[["open", "high", "close"]].min(axis=1))
    valid &= frame.segment_id.notna()
    if "quality_status" in frame:
        valid &= frame.quality_status.eq("ok")
    if not valid.iloc[-1]:
        result["reason"] = "latest_bar_invalid"
        return result
    start = len(frame) - 1
    while start > 0:
        current, previous = frame.iloc[start], frame.iloc[start - 1]
        if (not valid.iloc[start - 1] or current.segment_id != previous.segment_id or
                current.timestamp - previous.timestamp != pd.Timedelta(minutes=5)):
            break
        start -= 1
    block = values.iloc[start:]
    window = block.tail(6)
    prior = block.iloc[:-1].tail(20).volume
    stamps = frame.timestamp.iloc[start:]
    first, last = window.iloc[0], window.iloc[-1]
    result.update(status="ready" if len(window) == 6 else "partial", reason="observed_prefix_only",
                  window_count=len(window), window_start=stamps.iloc[-len(window)].isoformat(), window_end=stamps.iloc[-1].isoformat(),
                  price_change_pct=float(100 * (last.close / first.open - 1)),
                  range_pct=float(100 * (window.high.max() - window.low.min()) / first.open),
                  current_body_pct=float(100 * (last.close / last.open - 1)),
                  volume_reference_count=len(prior), boundary_limited=start > 0)
    if len(prior):
        result.update(volume_reference_start=stamps.iloc[-len(prior)-1].isoformat(), volume_reference_end=stamps.iloc[-2].isoformat())
    if len(prior) >= 6:
        median = float(prior.median())
        result.update(volume_reference_median=median, volume_reason="zero_reference_median" if median == 0 else "ready")
        if median > 0:
            result["volume_ratio"] = float(last.volume / median)
    return result


def summary_text(value):
    if value["status"] == "unavailable":
        return ["Chưa mô tả được: " + ("nến cuối không hợp lệ; kiểm tra chất lượng dữ liệu." if value["reason"] == "latest_bar_invalid" else "chưa có nến đã biết trong phạm vi này.")]
    change = value["price_change_pct"]
    movement = "cao hơn" if change > 0 else "thấp hơn" if change < 0 else "bằng"
    lines = [f"Trong {value['window_count']} nến liên tiếp, giá đóng cuối {movement} giá mở đầu {abs(change):.4f}%. "
             f"Khoảng cao–thấp của nhóm bằng {value['range_pct']:.4f}% giá mở đầu; đây là biên độ quan sát, không phải dự báo."]
    if value["volume_ratio"] is not None:
        lines.append(f"Volume nến cuối bằng {value['volume_ratio']:.2f} lần trung vị của {value['volume_reference_count']} nến trước đó trong cùng đoạn. "
                     "Nến đang xem không nằm trong mẫu tham chiếu; chưa điều chỉnh đặc thù giờ giao dịch.")
    else:
        reason = "trung vị tham chiếu bằng 0" if value["volume_reason"] == "zero_reference_median" else f"mới có {value['volume_reference_count']} nến trước, cần ít nhất 6"
        lines.append("Chưa so sánh volume: " + reason + ". Không thay giá trị thiếu bằng 0.")
    if value["window_count"] < 6 or value["boundary_limited"]:
        lines.append("Chỉ dùng phần liên tiếp hợp lệ đã biết; không nối qua gap, nến lỗi, nghỉ phiên hoặc lấy thêm nến tương lai.")
    return lines


def explain_method(event, detail, *, model=None):
    """Evidence only: never turn correlated bar descriptors into score attribution."""
    if event["kind"] == "insufficient_data":
        return ["Chưa đủ dữ liệu để phương pháp đánh giá; phần mô tả nến vẫn có thể đọc được. Không coi thiếu score là bình thường."]
    if model is None:
        return ["Phương pháp v2 so sánh điểm/đoạn với mẫu giả lập đã ghim. Hạng điểm không phải xác suất tăng giá; "
                "mô tả giá/volume bên trên không phải attribution của detector."]
    lines = ["Bản ghi v1 đã đánh dấu cửa sổ này." if event["kind"] == "point" else "Bản ghi v1 chưa đánh dấu cửa sổ này theo ngưỡng đang ghim."]
    if model != "lstm":
        lines.append("Chưa có attribution theo từng feature cho phương pháp này. Giá/volume mô tả bối cảnh, không đủ để kết luận feature nào làm score tăng.")
        return lines
    shares = {name: detail.get("share_" + name) for name in FEATURES}
    if (any(not isinstance(v, (int, float, np.number)) or isinstance(v, (bool, np.bool_)) or not math.isfinite(v) or not 0 <= v <= 1 for v in shares.values()) or
            not math.isclose(sum(shares.values()), 1, abs_tol=1e-5)):
        lines.append("Chưa có tỷ trọng lỗi tái dựng hợp lệ để giải thích feature; không tự tạo tỷ lệ đóng góp.")
        return lines
    largest = max(shares.values())
    names = ", ".join(FEATURES[k] for k, v in shares.items() if v == largest)
    lines.append(f"Phần lỗi tái dựng lớn nhất thuộc {names}: {100 * largest:.2f}% tổng lỗi của cửa sổ, theo tỷ trọng đã lưu. "
                 "Không phải xác suất thị trường, nguyên nhân kinh tế hay đóng góp riêng của nến cuối.")
    return lines
