"""Display-only explanations. Never change scores, missing values or event versions."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .theme import CHART, apply_chart_theme

KINDS = {"point": "Điểm cần xem", "collective": "Đoạn cần xem", "normal": "Chưa được đánh dấu",
         "insufficient_data": "Chưa đủ dữ liệu"}
STAGES = {"initial": "Đánh dấu ban đầu", "post_event": "Đã bổ sung kết quả"}
OUTCOMES = {"pending": "Đang chờ nến đích", "mature": "Đã có kết quả"}
LABELS = {"uncertain": "Chưa kết luận", "noteworthy": "Đáng nghiên cứu", "ordinary": "Không đáng chú ý",
          "data_issue": "Vấn đề dữ liệu"}


def local_time(value):
    return pd.Timestamp(value).tz_convert("Asia/Ho_Chi_Minh").strftime("%H:%M:%S")


def number(value):
    return "chưa đủ dữ liệu" if value is None or pd.isna(value) else f"{value:.4f}"


def explain(event, detail):
    kind = event["kind"]
    descriptions = {
        "point": "Một nến được phương pháp điểm đánh dấu để bạn xem lại, không phải tín hiệu mua hoặc bán.",
        "collective": "Một nhóm nến liên tiếp được phương pháp đoạn đánh dấu. Ranh giới chỉ dùng các nến đã biết lúc đánh dấu.",
        "normal": "Các phương pháp chưa đánh dấu nến này theo ngưỡng đang ghim. Điều đó không chứng minh nến bình thường.",
        "insufficient_data": "Chưa đủ nến hoặc mẫu tham chiếu để đánh giá. Đây không phải lỗi tải và không được coi là bình thường.",
    }
    methods = "So sánh một nến và các đoạn kết thúc tại nến này với mẫu tham chiếu tổng hợp đã ghim."
    if kind == "point":
        methods = "Phương pháp điểm: so sánh mức khác biệt của một nến với mẫu tham chiếu tổng hợp theo mã."
    elif kind == "collective":
        methods = "Phương pháp đoạn (suffix): so sánh các độ dài đoạn đã có, kết thúc tại nến đang xem."
    return {"title": KINDS[kind], "observation": descriptions[kind], "method": methods,
            "evidence": f"Hạng điểm: {number(detail['point_rank'])}; hạng đoạn: {number(detail['joint_rank'])}. Hạng không phải xác suất.",
            "missing": "Dữ liệu là giả lập; chưa có nhãn chuyên gia cho thử nghiệm này, chưa kiểm chứng hiệu quả thị trường thật. Tin đi kèm không chứng minh nguyên nhân."}


def event_table(events):
    return pd.DataFrame([{"Thời điểm (UTC+7)": local_time(row["event_end"]),
                          "Trạng thái": KINDS[row["kind"]], "Giai đoạn": STAGES[row["stage"]],
                          "Phiên bản": row["version"], "Hạng (không phải xác suất)": row["score"]} for row in events])


def timeline_figure(bars, events):
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.06, row_heights=[.75, .25])
    if not bars.empty:
        for index, (_, segment) in enumerate(bars.groupby("segment_id", sort=False)):
            figure.add_trace(go.Candlestick(x=segment.timestamp.dt.tz_convert("Asia/Ho_Chi_Minh"),
                open=segment.open, high=segment.high, low=segment.low, close=segment.close,
                name="Nến OHLC giả lập", showlegend=index == 0, legendgroup="price",
                increasing_line_color=CHART["up"], increasing_fillcolor=CHART["up"], increasing_line_width=1.5,
                decreasing_line_color=CHART["down"], decreasing_fillcolor=CHART["down"], decreasing_line_width=1.5), row=1, col=1)
        figure.add_trace(go.Bar(x=bars.timestamp.dt.tz_convert("Asia/Ho_Chi_Minh"), y=bars.volume,
                               name="Khối lượng giả lập", marker_color=CHART["volume"]), row=2, col=1)
        for event in events:
            if event["kind"] == "collective":
                figure.add_vrect(x0=pd.Timestamp(event["event_start"]).tz_convert("Asia/Ho_Chi_Minh"),
                                x1=pd.Timestamp(event["event_end"]).tz_convert("Asia/Ho_Chi_Minh"),
                                fillcolor=CHART["collective"], opacity=.16, line_width=0, row=1, col=1)
            elif event["kind"] == "point":
                match = bars.loc[bars.timestamp.eq(pd.Timestamp(event["event_end"]))]
                figure.add_trace(go.Scatter(x=match.timestamp.dt.tz_convert("Asia/Ho_Chi_Minh"), y=match.close,
                    mode="markers", name="Điểm cần xem", showlegend=False,
                    marker={"color": CHART["point"], "size": 10, "symbol": "diamond"}), row=1, col=1)
        if len(bars) == 1:
            clock = bars.timestamp.iloc[0].tz_convert("Asia/Ho_Chi_Minh")
            figure.update_xaxes(range=[clock - pd.Timedelta(minutes=5), clock + pd.Timedelta(minutes=5)])
    figure.update_layout(height=480, xaxis_rangeslider_visible=False, margin={"l": 15, "r": 15, "t": 10, "b": 20},
                         hovermode="x unified", legend={"orientation": "h"})
    figure.update_yaxes(title_text="Giá giả lập", row=1, col=1)
    figure.update_yaxes(title_text="Khối lượng", row=2, col=1)
    figure.update_xaxes(tickformat="%H:%M")
    return apply_chart_theme(figure)
