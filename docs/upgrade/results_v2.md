# Kết quả và cách hiểu bản nâng cấp

Bản nến v2 bổ sung suffix detector3/6/12 độc lập pointflags, heuristicselector,
forecast h1/3 lưu trước outcome, append-onlySQLite, causalreplayT, review vàCSV.
Phần mềm hoạt động không đồng nghĩa phát hiện tốt trên thị trường.

## U09 — benchmark tổng hợp chung, không tune

Seeds31/32 đã được xem từ U04, không phải holdout mới. Mỗi seed có1437endpoint
được cung cấp; intersection tất cả model/reference/variant là987 (68,69%).
Không bỏ pointpositive hoặc collectivetruth nào: mỗiseed9pointpositive và12đoạn.
Pointtask có906bars/seed, loại81bars thuộc collectivetruth và dataissue riêng.
Normal ngoài injection là nhãn mặc định theo spec đã khóa, không nhãn AI/human.
V1 model/scaler/threshold giữ nguyên từ dữ liệu thật, nên đây là out-of-domain.

Pooled pointcounts, cùng1812bars và18pointpositives:

|Phương pháp|TP|FP|FN|F1|
|---|---:|---:|---:|---:|
|V1 robust z-score|18|0|0|1,000|
|V1 Isolation Forest|18|0|0|1,000|
|V1 LSTM|13|12|5|0,605|
|Context robust rank|18|23|0|0,610|
|Context IF rank|18|32|0|0,529|
|Context LSTM rank|18|117|0|0,235|
|V2 suffix selector point|18|36|0|0,500|
|CAPA retrospective point|18|6|0|0,857|

Không kết luận baselineaccuracy thị trường100% từ18spikes giả lập. Với pointtask
này nên giữ robust/IF đối chứng; v2/context không cho cải thiện chung.

Pooled collective matching, IoU≥0,3 (sensitivity0,5 cùng kết quả),24truth:

|Phương pháp|TP|FP|FN|F1|
|---|---:|---:|---:|---:|
|V2 guarded multiscale|24|267|0|0,152|
|Fixed rolling6|24|300|0|0,138|
|No-single-point-guard|24|424|0|0,102|
|CAPA retrospective|24|3|0|0,941|

Mỗi truth chỉ ghép một prediction. Nhiều suffixemissions cùng hit truth vẫn
để nguyên, không oraclemerge để tăngF1; do vậy burden lớn. V2 có6,27/6,60
emissions mỗi mã/phiên trêncommoncoverage của hai seeds. CAPA xem toàn phiên,
không cùng informationbudget online, nên F1 cao không chứng minh thay thế realtime.
V1 làpointmethod, segmentmetrics N/A, không biến group-alerts thành detectorđoạn.

Boundaryerror0 của v2 trong bảng matching là **best matched emission**, không
firstalert. Reconciliation bổ sung firsthitdescriptive: firststart error trung
bình−0,292nến, firstend−3,292nến (ongoing suffix chưa thấy finalend). Logical
truthstart→firstdecision302–1502giây là clockfixture, không provider/live latency.
Actualoffline auditprocess362,02giây, gồm revision/delay/prefix kiểm tra; không
coi nó là thời gian chấm mỗi nến. Source→emit vàreceive→score live=N/A.

## Ablation, forecast và bối cảnh

No-context vàno-post giữ initial quyết định/metrics giốngv2base đúng thiết kế:
context làexplanation/rank riêng, chưa đi vào selector; post-evidence không sửa
initialclassification. No-segment có0TP/0FP/24FN đoạn: precision/F1 unavailable,
không giảprecision0 hoặc biến pointflags thành segment. No-guard biến spikes
thànhcollective, pointTP0 vàburden tăng; không đổi thresholds sau kết quả này.

Actualcontext references đều fallbacksymbol (990ranked/seed/model), directcontext0;
không chứng minh độ hiếm có điều kiện hiệu quả khi nhóm ít mẫu. U06 thử một
Ridge(alpha1), pooledvalidation gain~0,489% dưới gate1% đã chốt nên giữzero-return.
Mỗi heldseed h1:1434forecast/1371mature/60boundary/3gap/3lateoriginskip;
h3:1437forecast/1248mature/180boundary/9gap. Residualrank chỉ cho mature;
delayedF1 N/A vì chưa có rubric tái phân loại post-event, không nâng initialF1.

## Dữ liệu thật và giới hạn

V1 đã có486labels AI-assisted trên subset nămngày, đã xem: không independent
holdout và không chuyên gia. V2 chưa có blindsegmentlabels thật, tick/orderbook,
quyền nguồn confirmed hoặc latencylive mới. `implementation_met` và
`effectiveness_supported` là hai kết luận riêng: benchmark này chỉ hỗ trợ
implementation; effectiveness trên thị trường=false/unavailable.

Khuyến nghị giữ v1baseline, dùngv2 như công cụ ràsoát nghiên cứu. Nếu nâng chất
lượng tiếp: protocol mới với episode/dedup rule chốt trước, context có đủ mẫu,
nhãn đoạn độc lập và phiên source được xác minh. Không retune cùng test đã xem.

Logs/CSV/plots/hash/timing đầy đủ chỉ giữ trong workspace local tại run
U09_audit_ef0d3a4b8ee9, reconciliation và milestoneacceptance; không đóng gói
transcript/model/raw thật trong bản xuất. Xem limitation_matrix trước mọi claim
demo, publication hoặc production. Bản xuất local không phải deploy/push.
