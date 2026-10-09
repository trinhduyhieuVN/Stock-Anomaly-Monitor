# Demo5phút — chỉ dữ liệu tổng hợp

Chuẩn bị: làm theo README_WINDOWS, chạy app_upgrade.py ở localhost. Nói rõ
đây là bản nghiên cứu ở mức nến, không tick/LOB, không LIVE hoặc tín hiệu giao dịch.
Chọn FPT; fixture cùng tình huống cho HPG/VCB, không giả làm dữ liệu các mã thật.

| Thời lượng | Thao tác | Điều cần giải thích |
|---|---|---|
|0:00–0:40|Ngày2025-02-03, Replay tạiT, Tđầu|Chỉ nến đã biết tạiT; insufficient_data ban đầu không phải nến bình thường giả|
|0:40–1:30|Step tới nến index18; lọc point/initial; chọn event|Một spike riêng, lý do/rank/reference count; rank không phải xác suất|
|1:30–2:20|Ngày2025-02-04 hoặc06, tăngT qua vùng giữa buổi sáng; lọc collective|Chuỗi lệch nhẹ được quét độc lập3/6/12; suffix là ước lượng hiện tại, không phải finalend nhìn tương lai|
|2:20–3:00|Ngày2025-02-10, tab version/tin, soTtrước và sau tin known_at|Tin biết muộn chỉ xuất hiện sau availability; không chứng minh quan hệ nhân quả hoặc xóa nhãn|
|3:00–3:50|Ngày2025-02-13, từ event index18 sang các nến sau; tab forecast/outcome|Dự báo đã lưu trước target; thiếu outcome là pending, khi mature thêm version, initial không bị sửa|
|3:50–4:30|Ngày2025-02-14 cuối chiều và resetT|Cuối phiên censored, không residual0 hay nối ngày sau; reset không thấy future versions|
|4:30–5:00|Rà soát hiện tại, review thử với tên/reason rõ; filterreviewed và tảiCSV|Review dùng giờ thực; quay Replay2025 không thấy review mới. CSV đúng bộ lọc; test review không phải nhãn chuyên gia|

Context là lagged price/relative-volume proxy. Nhóm context ít mẫu fallback theo
mã; context chưa đi vào selectorU04. Demo không nạp ba model thật v1; chúng là
đối chứng local trong audit. Không quảng cáo performance thị trường từ fixture.
Nếu ngày/mã bị đổi thì cursor reset; nếu bộ lọc rỗng đọc hướng dẫn phục hồi,
không dựng số liệu khác để khiến demo trông đẹp.
