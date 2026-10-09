# Stock Anomaly Monitor

Ứng dụng rà soát bất thường giá và khối lượng cổ phiếu. Demo v2 có nến OHLC 5 phút, volume, cảnh báo điểm/đoạn, replay, forecast, ghi chú rà soát và xuất CSV. Mã nguồn nghiên cứu v1 với Robust z-score, Isolation Forest và LSTM Autoencoder vẫn được giữ trong repo.

## Chạy dashboard demo v2

Cần Python 3.12 trên Windows. Mở PowerShell:

```powershell
git clone https://github.com/trinhduyhieuVN/Stock-Anomaly-Monitor.git
cd Stock-Anomaly-Monitor
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e . --no-build-isolation --no-deps
.venv/Scripts/python.exe -m scripts.check_upgrade_demo
.venv/Scripts/python.exe -m scripts.run_upgrade_demo_local
```

Mở http://127.0.0.1:8502. Demo mặc định hiển thị toàn phiên; chọn **Replay tại T**
để xem từng bước. Nếu đang ở nến đầu, bấm **Xem toàn phiên**. Ô thiếu có dấu **—**
và lý do cụ thể. Bản demo dùng dữ liệu tổng hợp, không phải giá thị trường thật.

Launcher tự dừng sau 45 phút. Có thể dùng `--minutes 10` hoặc `--port 8503` khi
cổng mặc định đang bận. Muốn chạy không giới hạn thời gian:

```powershell
.venv/Scripts/python.exe -m streamlit run app_upgrade.py --server.address=127.0.0.1 --server.port=8502 --browser.gatherUsageStats=false
```

Xem [hướng dẫn demo](docs/upgrade/demo_local_hoan_thien.md) và
[kết quả, giới hạn](docs/upgrade/results_v2.md).

## Phạm vi bản public

Repository này là **bản xuất mã nguồn**, không phải bản sao dữ liệu nghiên cứu trên máy tác giả. Không chứa raw thị trường, dataset thực, observed calendar, model/scaler/threshold đã huấn luyện, báo cáo phiên, thông tin đăng nhập hoặc bộ điều phối cá nhân.

`demo/upgrade_ui/` chứa dữ liệu và calibration tổng hợp đã ghim hash để chạy dashboard v2 độc lập. Fixture nhỏ `demo/synthetic_ohlcv.csv` gồm 18 nến vẫn có thể kiểm bằng thư viện chuẩn Python. Giấy phép fixture không cấp quyền đối với dữ liệu thị trường hoặc mã nguồn; repository chưa thêm giấy phép mã nguồn riêng.

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
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e . --no-build-isolation --no-deps
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
app_upgrade.py         Dashboard demo v2 với dữ liệu tổng hợp đi kèm
src/stock_monitor/     Dữ liệu, feature, model, scoring, replay, realtime guards
tests/                 Nhóm tests độc lập của bản public
scripts/               Kiểm tra fixture và manifest
demo/                  Fixture tổng hợp + manifest phân phối
demo/upgrade_ui/        Dữ liệu và calibration cho dashboard v2
configs/default.yaml   Cấu hình strict mẫu, không có chứng nhận nguồn
```

Không commit `.env`, secrets, dữ liệu thực, model, logs hoặc trạng thái cá nhân. Việc public mã nguồn không đồng nghĩa cấp quyền sử dụng/thu thập/phân phối dữ liệu của bên thứ ba.
