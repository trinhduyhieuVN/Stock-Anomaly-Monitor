# Stock Monitor

Ứng dụng nghiên cứu những biến động khác thường của giá và khối lượng cổ phiếu trên nến 5 phút. Chọn mã, phát lại phiên giao dịch, xem tín hiệu và ghi nhận xét ngay trên biểu đồ.

## Chạy demo

Cần Windows x64 và Python 3.12. Mở PowerShell:

```powershell
git clone https://github.com/trinhduyhieuVN/Stock-Anomaly-Monitor.git
cd Stock-Anomaly-Monitor
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e . --no-build-isolation --no-deps
.venv/Scripts/python.exe -m scripts.run_demo
```

Mở **http://127.0.0.1:8515**. Những lần sau chỉ cần chạy lệnh cuối. Demo tự dừng sau 60 phút; `Ctrl+C` để dừng sớm, hoặc thêm `--port 8516` nếu cổng đang bận.

Bản trên GitHub chạy bằng dữ liệu giả lập đi kèm. Không cần tài khoản hay API dữ liệu. Dữ liệu thị trường và model đã huấn luyện không nằm trong repo.

## Có gì trong ứng dụng?

- Biểu đồ nến và khối lượng của FPT, HPG, VCB.
- Replay mượt: phát, tạm dừng, bước từng nến hoặc kéo thời điểm xem.
- Tách biến động theo quy tắc khỏi tín hiệu mô hình; xem chi tiết, ghi nhận xét và xuất CSV.
- Khi có bộ dữ liệu/model local hợp lệ: so sánh Robust z-score, Isolation Forest, LSTM Autoencoder và mô hình GRU học bối cảnh trước nến hiện tại.

## AI hoạt động thế nào?

Nến → đặc trưng giá/khối lượng → model đã huấn luyện → mức bất thường → tín hiệu để rà soát.

LSTM tìm mẫu khó tái tạo. GRU học phân bố có điều kiện từ 12 nến trước; bộ chọn tiếp theo phân biệt biến động đơn lẻ và đoạn nhiều nến. Model, scaler và ngưỡng giữ nguyên khi replay. Phần dự báo ngắn hạn hiện dùng baseline giá không đổi vì Ridge chưa vượt baseline trên validation.

Đây là demo nghiên cứu, chưa có đánh giá độc lập với nhãn chuyên gia cho mô hình mới. Cảnh báo không phải kết luận thao túng hoặc lệnh mua/bán; replay không phải dữ liệu trực tiếp.

[Hướng dẫn demo](docs/huong_dan_demo_local.md) · [Báo cáo ngắn](docs/bao_cao_du_an.md)

## Kiểm thử

```powershell
.venv/Scripts/python.exe -m pytest -q
```

Mã xử lý nằm trong `src/stock_monitor/`, giao diện trong `web/research-workbench-replay/public/`, dữ liệu giả lập trong `demo/upgrade_ui/`. Bản public không chứa dữ liệu thật, trọng số đã huấn luyện, báo cáo phiên, nhật ký cá nhân hoặc thông tin đăng nhập. Repo chưa có giấy phép mã nguồn riêng.
