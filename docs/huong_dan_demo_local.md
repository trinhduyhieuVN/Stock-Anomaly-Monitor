# Demo Stock Monitor

Sau khi cài theo [README](../README.md), mở PowerShell tại thư mục dự án:

```powershell
.venv/Scripts/python.exe -m scripts.run_demo
```

Mở **http://127.0.0.1:8515**. Giữ terminal mở trong lúc dùng; `Ctrl+C` để dừng. Cổng đang bận thì thêm `--port 8516`, không cần tắt ứng dụng khác.

## Xem thử trong 3 phút

1. Chọn dữ liệu giả lập, mã **FPT**, ngày **03/02/2025**. Xem nến và khối lượng của cả phiên.
2. Đưa thanh Replay về đầu phiên rồi bấm **Phát Replay**. Nến xuất hiện lần lượt, không tải lại trang. Tạm dừng để xem phần chi tiết theo thời điểm đang chọn.
3. Kéo đến **10:35** để xem biến động giá; sau **13:05** xem thay đổi khối lượng. Phần biến động quan sát dùng quy tắc minh họa, không phải kết quả AI trên thị trường.
4. Mở tín hiệu, xem điểm/đoạn liên quan, ghi nhận xét nếu có và xuất CSV. Replay chỉ xuất dữ liệu đã biết đến thời điểm chọn.

## Dữ liệu thật và AI

Bản GitHub không kèm dữ liệu thật hay trọng số model. Trên máy có bộ nghiên cứu local đã được kiểm chứng, giao diện tự nhận các manifest tương ứng và mở thêm nguồn lịch sử cùng danh sách model. Không tự tải dữ liệu hoặc huấn luyện khi mở trang.

Với bộ AI local hiện tại, có thể xem **VCB / 17/09/2026 / AI · điểm / đoạn**, đến **10:25**; hoặc **HPG / 15/09/2026**, đến **11:00** để xem ví dụ điểm và đoạn. Đây là ví dụ lịch sử đã khảo sát, không phải đánh giá độc lập. Ngày trước mốc đóng băng hoặc nến thiếu đầu vào có thể chưa có score.

## Đọc tín hiệu

- **Biến động quan sát:** thay đổi giá/volume theo quy tắc, giúp đọc biểu đồ.
- **Tín hiệu mô hình:** mức khác thường theo model đã chọn, không phải xác suất giá tăng.
- **Chưa đủ dữ liệu / đang chờ:** thiếu lịch sử, có gap hoặc chưa đến lúc đánh giá kết quả; không được hiểu thành score bằng 0.
- **Dự báo:** baseline ngắn hạn; kết quả chỉ được bổ sung khi các nến tiếp theo xuất hiện.

Nếu chưa thấy tín hiệu, kiểm tra nguồn, mã, ngày và model. Không phải phiên nào cũng có cảnh báo. Bản giả lập minh họa cách dùng; muốn nghiên cứu thị trường cần dữ liệu có quyền sử dụng và model phù hợp.
