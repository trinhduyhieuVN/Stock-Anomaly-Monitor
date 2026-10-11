# Stock Monitor — báo cáo ngắn

Cập nhật 11/10/2026.

## Mục tiêu

Giúp người nghiên cứu tìm và kiểm tra những biến động giá–khối lượng khác thường trên nến 5 phút. Ứng dụng hỗ trợ quan sát và rà soát, không xác định người thao túng hay tự giao dịch.

## Đã làm được

Giao diện local có nến, volume, replay mượt, tín hiệu điểm/đoạn, giải thích, ghi nhận xét và xuất CSV. Bản GitHub có dữ liệu giả lập để chạy thử ngay; bộ nghiên cứu thật giữ riêng trên máy.

Pipeline cũ gồm kiểm dữ liệu, đặc trưng nhân quả, Robust z-score, Isolation Forest và LSTM Autoencoder. Bộ dữ liệu lịch sử local có 16.907 nến của FPT, HPG, VCB trên 126 ngày chung. Dữ liệu thiếu không được tự điền; model/scaler fit train, ngưỡng lấy từ validation.

Phần nâng cấp thêm GRU học bối cảnh 12 nến trước, phân bố Student-t và bộ chọn ba cách giải thích: bình thường, một điểm khác thường hoặc một đoạn khác thường. Quyết định ban đầu được giữ; kết quả sau 1/3 nến được bổ sung thành phiên bản mới. Ridge dự báo chưa vượt baseline trên validation nên demo giữ dự báo giá không đổi.

## Kết quả hiện tại

Lần kiểm thử toàn hệ thống đã lưu đạt **1.026 tests**, không lỗi; giao diện có 12 kiểm tra trình duyệt đạt và bản sao sạch chạy được với dependency đã khóa. Đây là kiểm chứng phần mềm, không phải độ chính xác AI.

Đánh giá cũ trên 486 điểm có nhãn AI-assisted cho recall khoảng 0,8–1,6%, chưa phải nhãn chuyên gia. Mô hình GRU mới chưa có tập test độc lập và nhãn người rà soát để kết luận hiệu quả; chưa chọn model thắng cuộc.

## Còn thiếu

- Xác nhận quyền dữ liệu, đơn vị và lịch phiên trước khi phân phối dữ liệu thật.
- Dữ liệu mới chưa từng dùng để phát triển và nhãn chuyên gia để đánh giá AI.
- Hoàn thiện kết nối đánh giá theo thời điểm nhận dữ liệu và thử nhiều phiên thực tế. Demo replay không chứng minh khả năng chạy live liên tục.
- Dữ liệu từng giao dịch/order book nếu muốn nghiên cứu dòng lệnh hoặc khoảng thời gian giữa giao dịch; OHLCV hiện tại chưa đáp ứng phần đó.

Sản phẩm hiện phù hợp để demo quy trình và tiếp tục nghiên cứu. Chưa đủ bằng chứng để dùng làm hệ thống quyết định đầu tư.

[Chạy và trình bày demo](huong_dan_demo_local.md)
