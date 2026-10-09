# FAQ và checklist bản thu gọn

## Câu hỏi thường gặp

**Dự án làm gì?** Ưu tiên rà soát nến bất thường riêng lẻ hoặc một đoạn nến,
giải thích reference/bối cảnh quá khứ, lưu forecast/outcome và review có version.
Không xác định tài khoản/thao túng hoặc tự giao dịch.

**Vì sao không có LIVE?** Bundle này là tổng hợp với lịch/độ trễ logic. Trial
v1 cũ chỉ chứng minh các phiên đã lưu; không phải feed đang chạy hôm nay.
VCB và quyền nguồn còn giới hạn độc lập. Không bật provider để làm demo.

**Rank0.99 nghĩa gì?** Mức hiếm thực nghiệm trong tập tham chiếu, không99%
gian lận. Ít mẫu thì fallback hoặc unavailable; không tự giảm minimum100.

**Context có tác dụng chọn đoạn chưa?** Hiện lớp giải thích và point ranking
tách khỏi suffix selector. No-context cho initialU04 giống nhau; đây là phần
tích hợp chưa có, không phải kết luận khoa học rằng context không cần thiết.

**Forecast có nhìn tương lai?** Prediction được đóng băng trước target. Outcome
chỉ dùng khi đủ cả path1/3nến; trễ/gap/cuối phiên pending/censored, không điền0.
Post-event thêm version, không nâng quality của cảnh báo ban đầu bằng thông tin sau.

**CAPA có phải tên khác của heuristic?** Không. Audit dùng sktime1.2.0 official
trong envriêng; batch toàn phiên là retrospective. UI dùng heuristic riêng3/6/12,
không gọi nó làCAPA hay posterior/likelihood.

**Lỗi manifest/missing resource?** Chạy check_upgrade_demo.py. Khôi phục đúng
bundle/config từ bản xuất đã kiểm, không upload joblib/pickle tùy ý hoặc sửahash.

**Lỗi cổng/khóa SQLite?** Chọn cổng khác nếu appkhác đang chạy. Writer khác đang
hoạt động phải được xác định và chọn phiên tường minh, không xóa lock/killbừa.
Journal có bounded busy timeout; chờ thao tác writer kết thúc rồi thử lại.

**Reset có xóa review?** Không. Reset chỉ đổi as-of. Review hôm nay thuộc giờ thực
và chế độ hiện tại; không backdate vào replay để giả nhãn đã biết trước.

**Có đẩy GitHub/deploy không?** Lượt này chỉ chuẩn bị và kiểm bản xuất local,
không push/deploy. Không đem nguồn thật/model/session/personalDOCX ra public.

## Checklist demo

- Python3.12/venv riêng; check_environment và pipcheckPASS.
- Manifest/config/dataset/calibration hashPASS; không realartifacts.
- Hiện SYNTHETIC/not-LIVE và disclaimer rà soát, không xác suất/gian lận.
- Point/collective/known-laternews/pending→mature/censored đúng as-of.
- CSV khớp filters và version; review lưu rõreviewer_kind/tên/reason.
- AppTest/localhost kiểm thực tế; không coi HTTP200 là fullUX/latency.
- Giữ kết quả benchmark xấu, missing/exclusions và limitations.
- Không provider/cost/trading/publication hoặc sửa v1/coordinator.
