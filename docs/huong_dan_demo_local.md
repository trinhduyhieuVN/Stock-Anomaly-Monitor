# Demo Stock Monitor

Sau khi cài theo [README](../README.md), mở PowerShell tại thư mục dự án:

```powershell
.venv/Scripts/python.exe -m scripts.run_demo
```

Mở **http://127.0.0.1:8515**. Giữ terminal mở trong lúc dùng; `Ctrl+C` để dừng. Cổng đang bận thì thêm `--port 8516`, không cần tắt ứng dụng khác.

## Xem thử trong 3 phút

1. Chọn dữ liệu giả lập, mã **FPT**, phiên **Dao động hai chiều · 17/02/2025** để xem đường giá lên xuống rõ. Phiên này mặc định khi có fixture mới. Muốn xem ví dụ cũ thì chọn **03/02/2025**; dữ liệu cũ không bị thay thế.
2. Chọn **Đường giá** hoặc **Nến** phía trên biểu đồ. Đưa thanh Replay về đầu phiên rồi bấm **Phát Replay**. Có thể đổi loại biểu đồ khi đang phát; thời điểm không reset và trang không tải lại. Tạm dừng để đọc score tại đúng mốc đó.
3. Chọn **12 nến / 24 nến / Cả phiên** để thay vùng nhìn; trục giá theo các nến trong vùng đó, không đổi giá hay mốc Replay. Chọn nến/sự kiện cũ thì vùng nhìn đưa nến đó vào. Phiên cũ03/02 có biến động giá lúc10:35 và volume sau13:05. Phần biến động quan sát dùng quy tắc mô tả, không phải kết quả AI trên thị trường.
4. Xem **Tín hiệu mô hình** bên phải để đọc kết quả, score và ngưỡng; dùng tab cùng tên để chọn điểm/đoạn. Mở **Chi tiết mô hình** nếu cần giải thích, hoặc xuất CSV. Replay chỉ xuất dữ liệu đã biết đến thời điểm chọn; các thẻ mã phía trên cũng không hiển thị giá tương lai. Bản giao diện này không có chức năng ghi nhận xét.

Chấm vàng ở chế độ quan sát chỉ giữ tối đa3 điểm nổi bật trong vùng đang xem,
cách nhau ít nhất3 nến. Danh sách bên dưới vẫn đầy đủ; bộ lọc này không thay score
hoặc chấm tín hiệu mô hình. Các điểm nổi bật có thể đổi khi Replay/zoom đổi phần đã biết.

## Dữ liệu thật và AI

Bản GitHub không kèm dữ liệu thật hay trọng số model. Trên máy có bộ nghiên cứu local đã được kiểm chứng, giao diện tự nhận các manifest tương ứng và mở thêm nguồn lịch sử cùng danh sách model. Không tự tải dữ liệu hoặc huấn luyện khi mở trang.

Phiên dao động mới là dữ liệu minh họa, không phải giá thật FPT/HPG/VCB. Score được tính lại từ nến mới bằng bộ chấm điểm giả lập đã đóng băng, không sao chép score cũ. Vì dao động mạnh hơn mẫu hiệu chuẩn, phiên này có thể có nhiều điểm được đánh dấu; không dùng nó để kết luận model chính xác hơn. Logo chỉ giúp nhận diện doanh nghiệp, không thể hiện bảo trợ.

Với bộ AI local hiện tại, có thể xem **VCB / 17/09/2026 / AI · điểm / đoạn**, đến **10:25**; hoặc **HPG / 15/09/2026**, đến **11:00** để xem ví dụ điểm và đoạn. Đây là ví dụ lịch sử đã khảo sát, không phải đánh giá độc lập. Ngày trước mốc đóng băng hoặc nến thiếu đầu vào có thể chưa có score.

## Đọc tín hiệu

- **Biến động quan sát:** thay đổi giá/volume theo quy tắc, giúp đọc biểu đồ.
- **Tín hiệu mô hình:** mức khác thường theo model đã chọn, không phải xác suất giá tăng.
- **Chưa đủ dữ liệu / đang chờ:** thiếu lịch sử, có gap hoặc chưa đến lúc đánh giá kết quả; không được hiểu thành score bằng 0.
- **Dự báo:** baseline ngắn hạn; kết quả chỉ được bổ sung khi các nến tiếp theo xuất hiện.

Nếu chưa thấy tín hiệu, kiểm tra nguồn, mã, ngày và model. Không phải phiên nào cũng có cảnh báo. Bản giả lập minh họa cách dùng; muốn nghiên cứu thị trường cần dữ liệu có quyền sử dụng và model phù hợp.
