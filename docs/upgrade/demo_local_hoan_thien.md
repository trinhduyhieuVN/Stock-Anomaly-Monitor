# Chạy và trình diễn demo local

Đây là dữ liệu **SYNTHETIC**, không phải LIVE hoặc dữ liệu thật của FPT/HPG/VCB.
Ứng dụng ưu tiên rà soát nến 5 phút, không kết luận gian lận hay đặt lệnh.
Detector, ngưỡng và kết quả benchmark đã khóa được giữ nguyên.

## Một lệnh chạy trên máy hiện tại

Mở PowerShell tại thư mục project, nơi có `app_upgrade.py` và venv đã cài:

```powershell
./.venv/Scripts/python.exe -X utf8 -m scripts.run_upgrade_demo_local
```

Mở **http://127.0.0.1:8502** sau khi launcher báo `READY`. Launcher kiểm manifest
trước khi chạy, chỉ nghe localhost, tắt telemetry và tự dừng sau 45 phút.
Ctrl+C trong chính cửa sổ launcher dừng server con của nó; không dừng phiên khác.
Nếu cổng bận, không xóa lock hay kill ứng dụng: dùng `--port 8503` và mở đúng URL.
Có thể chọn `--minutes 10`; tối đa 60 phút. `--check-only` chỉ kiểm tra, không
khởi động server. Log thật nằm ở `reports/demo_completion/local_<id>/`.

Để cài lại trên máy khác, dùng README_WINDOWS và bản xuất synthetic đã nghiệm
thu; launcher bổ sung này nằm trong workspace local, không tự cập nhật ZIP cũ.
Không tự tải model, gọi provider hay bật feed thật khi mở demo.

## Xem ngay biểu đồ nến và đọc ô trống

Demo local hiện mở ở **Rà soát hiện tại**: toàn phiên synthetic 48 nến cho ngày
03/02/2025. Biểu đồ dùng OHLC thật của fixture, nến xanh tăng/đỏ giảm, kèm volume.
Sự kiện cần xem ưu tiên điểm/đoạn đã phát hiện thay vì nến đầu chưa đủ lịch sử.
Nếu tab cũ còn giữ Replay tại T ở đầu phiên, bấm **Xem toàn phiên** hoặc chọn
**Rà soát hiện tại** trong sidebar. Replay vẫn bắt đầu ở một nến; đó không phải
lỗi tải. Đổi mã/ngày giữ lựa chọn chế độ, nhưng reset vị trí replay.

Dấu **—** chỉ thay cách hiển thị giá trị thiếu, không sửa dữ liệu hoặc CSV:
`insufficient_data` cần thêm lịch sử/tham chiếu; `pending` đang chờ nến đích;
`censored_boundary` không thể chấm kết quả qua ranh giới phiên/đoạn. Ứng viên
điểm và ứng viên đoạn nằm ở hai bảng riêng vì không dùng cùng các trường.
Chưa có tin hoặc rà soát được ghi rõ, không tạo bản ghi giả để lấp bảng.
Forecast zero-return cố ý dự đoán lợi suất bằng 0, **không** dự đoán giá bằng 0.

Bản sửa giao diện này chỉ áp dụng workspace hiện tại; ZIP/historical reports
U08/U10 cũ giữ nguyên nội dung và không đại diện cho source UI đã sửa.

## Sáu tình huống trong năm phút

Trong sidebar chọn **FPT**, **Replay tại T**. Đổi mã/ngày sẽ reset T. Thanh T
dùng giờ UTC+7; các bảng thô vẫn có timestamp UTC với offset, không đổi dữ liệu.

| Tình huống | Ngày và giờ T | Thao tác / kết quả cần thấy |
|---|---|---|
| Điểm riêng lẻ | 03/02/2025, 10:35:02 | Loại `point`, giai đoạn `initial`: một điểm, version1; forecast còn pending. Không gọi rank là xác suất. |
| Chuỗi lệch nhẹ | 06/02/2025, 11:00:02 | Loại `collective`, giai đoạn `initial`: vùng tím là suffix đã biết tại T, không phải final-end tương lai. |
| Tin biết muộn | 10/02/2025, 10:35:02 → 10:45:02 | Bật mọi loại/giai đoạn và tab Versions & tin: trước T chưa có tin, sau T mới có tin tổng hợp. Reset T phải ẩn tin. |
| Pending → mature | 13/02/2025, 10:35:02 → sau 10:55 | Chọn sự kiện endpoint10:35:00, tab Forecast: pending rồi mature; version mới nhưng kind/score ban đầu giữ nguyên. Quay T cũ không thấy outcome mới. |
| Cuối phiên | 14/02/2025, 14:30:02 | Chọn endpoint14:30:00: `censored_boundary`, residual thiếu chứ không phải0. Reset không nối dữ liệu sang hôm sau. |
| Review và CSV | 03/02/2025, Rà soát hiện tại | Chọn endpoint10:35, nhập tên/lý do rõ và lưu. Chọn `reviewed` để refresh bộ lọc rồi tải CSV. Quay Replay2025 không thấy review ghi hôm nay. |

Khi chuyển tình huống, đặt lại bộ lọc loại/giai đoạn và trạng thái rà soát `all`
nếu bảng rỗng. Không giảm ngưỡng để tạo cảnh báo. Review demo chỉ là QA, không
được dùng làm nhãn chuyên gia hoặc số liệu accuracy. Reset giữ nguyên journal.

## Những giới hạn phải nói khi trình diễn

Context hiện là proxy quá khứ, chưa tham gia selector. Demo độc lập không nạp
ba model thật v1; chúng được kiểm trong workspace bằng frozen artifacts. Nhiều
suffix chồng lấn vẫn được lưu: segment F1 synthetic0.15238 và267FP không bị
che hoặc tune lại. CAPA là đối chứng retrospective, không có trong luồng online.
Tick/sổ lệnh, quyền nguồn, nhãn expert và real holdout mới vẫn chưa đủ bằng chứng.

Các kiểm tra functional, AppTest, screenshot và HTTP200 không chứng minh
market performance, LIVE, WCAG toàn diện hay vận hành production.
