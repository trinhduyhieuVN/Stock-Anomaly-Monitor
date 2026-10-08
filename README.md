# Stock Anomaly Monitor

Ứng dụng nghiên cứu phát hiện bất thường giá/khối lượng cổ phiếu bằng Robust z-score, Isolation Forest và LSTM Autoencoder. Giao diện Streamlit tiếng Việt có biểu đồ, giải thích lỗi tái tạo, xuất CSV và replay nhân quả.

## Phạm vi bản public

Repository này là **bản xuất mã nguồn**, không phải bản sao dữ liệu nghiên cứu trên máy tác giả. Không chứa raw thị trường, dataset thực, observed calendar, model/scaler/threshold đã huấn luyện, báo cáo phiên, thông tin đăng nhập hoặc bộ điều phối cá nhân.

Fixture `demo/synthetic_ohlcv.csv` gồm 18 nến được tạo tổng hợp, không phải giá thật hoặc dự báo. Manifest công bố fixture theo CC0-1.0. Không suy ra quyền dữ liệu thị trường hoặc giấy phép mã nguồn từ giấy phép fixture; repository chưa cấp một giấy phép mã nguồn riêng.

## Chạy demo tổng hợp ngay (Python 3.12)

```powershell
git clone https://github.com/trinhduyhieuVN/Stock-Anomaly-Monitor.git
cd Stock-Anomaly-Monitor
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m scripts.run_synthetic_demo
```

Lệnh chỉ dùng thư viện chuẩn Python: kiểm SHA256/schema/OHLCV và in thống kê fixture. **Đây không phải demo suy luận của model đã huấn luyện và không mở dashboard.**

## Cài môi trường cho mã nguồn và tests

```powershell
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe -m pytest -q
```

`requirements-lock.txt` ghi phiên bản của môi trường Windows nghiên cứu gốc để đối chiếu; khả năng cài còn phụ thuộc hệ điều hành/index đang dùng. Bản public chỉ gồm nhóm tests độc lập dùng fixture tổng hợp; không chứa toàn bộ audit phụ thuộc dữ liệu local. Không có test tự gọi provider thật trong nhóm xuất này.

## Dashboard đầy đủ

`app.py` chứa giao diện nghiên cứu. Để chạy đầy đủ cần tự cung cấp dữ liệu có quyền sử dụng và các bundles feature/model/scaler/threshold/scores cùng provenance theo APIs trong `src/stock_monitor`. Artifacts private không nằm trong GitHub này.

```powershell
.venv/Scripts/python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Với một clone mới chưa có bundles, ứng dụng sẽ báo thiếu dữ liệu/model; đây là trạng thái dự kiến, không phải demo hoàn chỉnh. Không tự tải hoặc tự huấn luyện khi rerun giao diện. `configs/default.yaml` là ví dụ strict: xác nhận nguồn vẫn pending, các đường dẫn raw/calendar/evidence là đầu vào cần được cung cấp độc lập, không phải tài nguyên đính kèm.

## Phương pháp

Raw 1 phút → kiểm chất lượng → nến 5 phút → bốn feature nhân quả → cửa sổ sáu nến → score → ngưỡng validation → cảnh báo.

- Features: log return, range ratio, body ratio và volume surprise so với các phiên trước cùng khung giờ.
- Model/scaler fit train; threshold phân vị 98% validation theo mã. Không tune trên test.
- LSTM Autoencoder: học tái tạo cửa sổ 6×4; score là MSE bốn feature tại nến cuối.
- Replay chỉ cho thấy dữ liệu đến thời điểm T; không coi replay là live.
- Incremental xử lý idempotence/revision, raw bất biến, nến đã đóng, gaps/stale và khóa artifact; retry hữu hạn, fail-closed khi không đủ điều kiện.

Ngưỡng 98% **không phải độ chính xác 98%**. Cảnh báo không phải khuyến nghị mua/bán, bằng chứng thao túng hoặc cam kết sinh lợi. LSTM không mặc nhiên tốt hơn baseline. Nhãn AI-assisted hoặc benchmark synthetic không thay thế ground truth chuyên gia và đánh giá thị trường độc lập.

## Cấu trúc

```text
app.py                 Giao diện Streamlit
src/stock_monitor/     Dữ liệu, feature, model, scoring, replay, realtime guards
tests/                 Nhóm tests độc lập của bản public
scripts/               Kiểm tra fixture và manifest
demo/                  Fixture tổng hợp + manifest phân phối
configs/default.yaml   Cấu hình strict mẫu, không có chứng nhận nguồn
```

Không commit `.env`, secrets, dữ liệu thực, model, logs hoặc trạng thái cá nhân. Việc public mã nguồn không đồng nghĩa cấp quyền sử dụng/thu thập/phân phối dữ liệu của bên thứ ba.
