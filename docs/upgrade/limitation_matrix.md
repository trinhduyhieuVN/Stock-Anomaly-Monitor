# R01–R13: phạm vi đạt và giới hạn

Đây là bản nến5m/proxy. Hoàn thành coreU00–U10 không đồng nghĩa hoàn thành
toàn bộ đề cương tick/orderbook. Hiệu quả thị trường chưa được chứng minh.

|Yêu cầu|Trạng thái bản thu gọn|Bằng chứng/giới hạn|
|---|---|---|
|R01 từng giao dịch, tick/LOB|deferred|V1 có closed5m bounded trial; T01 không có tick/book/quyền để thử|
|R02 hiếm theo context|implemented/fallback|U03 trainbins/validationrefs/as-of; actualdirectcontext refs sparse, fallbacksymbol. Lớp ranking tách selector|
|R03 nghịch dòng/thanh khoản|proxy only|Lagged price trend/relativevolume; không aggressor/bidask/depth/orderflow|
|R04 đoạn đa độ dài|implemented bar-level|U04 riêng3/6/12, jointmaxvalidation, không nhóm pointalerts; nhiều emissions trùng gây burden|
|R05 sau sự kiện/dự báo|implemented synthetic|U06 h1/3 forecast-before-outcome, MAE/rank/pathcensor/revision; zero giữ vì Ridgegain dưới gate1%|
|R06 UI/context/review|implemented local|U08 causalT/filter/tab/CSV/appendreview; initial desktop/mobile viewportQA, không toànUX/WCAG certification|
|R07 density deep contextual|deferred/contrast only|V1 frozenLSTM6×4 đối chứng local, không mạng density mới hoặc fraud probability|
|R08 thứ tự/interarrival|bar proxy only|Chuỗi5m và thời điểm availability; không gọi5m là khoảng cách giao dịch tick|
|R09 selector point/segment/normal|heuristic implemented|U04 penalty/influence/coherence guard; không mô hình Bayesian/likelihood. Độ hiệu quả khác implementation|
|R10 IF/CAPA|implemented audit|V1frozenIF; officialCAPAmeanL2 batchenvriêng, retrospective không online latency; notUICAPA|
|R11 metrics/boundary/latency|offline/exploratory|U09 sharedsynthetic one-to-one/coverage/ablation; real486AI-assisted đã xem, newrealholdout/expert/live latency unavailable|
|R12 spike/mild/news|synthetic implemented|U02seeds cases và known_at; Journal API nhậnmetadata news đã validate. Không crawl tin hoặc claim nguyên nhân thật|
|R13 chỉ ưu tiên rà soát|implemented|UI/docs/contracts/report cấm fraud/trading/probabilityclaim; reviewer_kind tường minh|

Sáu source confirmations thật vẫn pending; không chuyển confirmed qua testPASS.
New real holdout, expert segmentlabels và quyền dữ liệu/phiên source tốt cần
bằng chứng ngoài phần mềm. T01 không mua/thu thập mới, không phải milestonecomplete.
V2synthetic fixture là distributable, raw/model/session thật vẫn local-only.
