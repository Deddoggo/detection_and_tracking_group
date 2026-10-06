# Checkpoint hệ thống — 05/10/2026

**Manual-annotation workflow:** tác giả xác nhận chưa có công cụ annotation và
muốn tạo GT trước, chưa inference. Đã chuẩn bị CVAT local workflow (chưa cài/chạy
vì Docker engine chưa hoạt động), clip15s/419 frames giữ nguyên decoded pixels,
mapping clip→source frame832–1250/timestamps, bbox+ID labels, native/dense CVAT
converter, human review checklist và group reference template riêng. Xem
`MANUAL_ANNOTATION_WORKFLOW_VI.md`. Không có GT hoàn chỉnh hay model inference.

**Cập nhật sau khi tác giả thêm videos/:** đã có một video raw 106.158688 giây,
1920×1080, 2.956 frame. Đã kiểm tra decoded timestamps: nhịp frame thay đổi,
FPS trung bình 27.848645 khác FPS danh nghĩa60. Công cụ FFmpeg/ffprobe đã được
tìm thấy trong môi trường rugby-backend; không cài thêm thư viện. Đã xem các frame
5/35/65/95s và lập ba candidate intervals 30–45s, 60–75s, 88–103s cùng manifest,
frame ranges, source descriptors, config và annotation pending. Cả ba cùng một
video, không độc lập. Chưa xem liên tục để xác nhận mọi crossing/occlusion và chưa
có predictions để xác nhận miss/FP/duplicate. Không chạy inference. Kiểm tra timing
mới đã pass; tổng **13 tests correctness**, không phải metrics trên GT. Phần dưới
giữ checkpoint ban đầu để phân biệt những thông tin đã được bổ sung.

## What we know — điều đã xác nhận

- Repository bắt đầu ở `master`, commit `2db7aac`, working tree sạch. Snapshot này
  sau thesis; không gán cấu hình hiện tại cho archived thesis results.
- Có nhiều đường chạy. Chuỗi playground offline trong code là YOLO26 → ByteTrack
  → stitching tùy chọn → ViTPose/orientation/social grouping. Chuỗi integrated khác
  dùng DeepOCSORT hoặc BoostTrack. Chưa biết lệnh thực tế gần nhất của tác giả.
- Cả hai module grouping đều có lỗi overwrite khi split; gate absolute cosine
  trong module social không chọn lọc vector 2D đơn vị hợp lệ.
- Không có thư mục videos/, detection/track JSON playground, checkpoint .pt hoặc
  môi trường inference trong checkout này. Video có sẵn chủ yếu là output/PETS;
  AVI dưới runs/ cần xác nhận nguồn. Chưa chọn đoạn raw hay timestamp nào.
- Thông tin mới do tác giả cung cấp: **5 video raw**, khoảng 2+2+20+20+30=74 phút;
  không có GT bbox/ID/orientation/group. Đây là mô tả của tác giả, chưa phải metadata
  đo từ các tệp raw.

## What we suspect — giả thuyết còn mở

- Misses/blur/occlusion/crossing có thể là bottleneck đầu vào; giảm confidence có
  thể cải thiện recall nhưng tăng FP/duplicates và làm association khó hơn.
- Ngưỡng cho recall detection cao nhất có thể không cho tracking/grouping tốt nhất.
- Tiny-box filtering trong analyzer, stitching và tracker configuration có thể
  ảnh hưởng đầu ra; post-construction buffer override có thể không có tác dụng
  mong muốn trong phiên bản BoxMOT thực tế. Chưa có source môi trường để xác nhận.
- Pose direction/geometry và membership có thể yếu độc lập với tracking. Chưa
  chọn detection/tracking, grouping/orientation hay robust grouping under imperfect
  detection/tracking làm hướng chính. Chưa xác nhận improvement đã bão hòa.

## What remains unmeasured — chưa đo

Detection precision/recall/FP/FN/duplicates trên trẻ em; tracking HOTA/DetA/AssA/
IDF1/IDSW/Frag trên playground; group membership/stability/split/merge; độ tin cậy
orientation; geometry scale; thời gian chạy và nguồn lực GPU. Không có metric mới
trên GT, không có quan sát video raw mới và không có claim improvement/novelty.

## What was implemented and verified — code đã hoàn thành

**A. Engineering correctness:** giữ full-source archive và hai grouping source gốc;
matching group ID một-một, unique IDs, split/merge policy rõ; không overwrite và
không giữ membership cũ khi một frame đã quan sát không có người. Đây là bug fix,
không phải research contribution hay bằng chứng membership accuracy.

**B. Infrastructure:** export/cache chuẩn pixel xywh, absolute frame index, score/
track ID; nhập JSON cũ; giữ cached keypoints/direction nếu đã có; replay grouping;
config/package/checkpoint/calibration/source hashes; inventory và manifest pending;
annotation interface; detection evaluator; confidence sweep; oracle input builder;
strict MOT export và chuẩn bị thư mục official TrackEval. Runner detector/tracker
giới hạn <=20 giây, không tự đổi checkpoint hoặc dùng YAML mặc định. ByteTrack
được tái sử dụng, không viết lại association/low-confidence recovery.

**C. Candidate:** giữ raw legacy gate, proximity-only comparator và signed-facing
experimental variant bật/tắt được. Candidate cần direction có dấu và reliability
đáng tin; thiếu cue/reliability thì fallback proximity. Chưa xem keypoint confidence
là heading reliability; chưa dùng candidate để kết luận social grouping cải thiện.

**12 kiểm tra correctness đã pass**, dùng fixtures tổng hợp có đáp án biết trước: tái hiện chính
assignment block trong hai source gốc và đối chiếu block sửa; split/merge/ties;
gate; maximum-cardinality IoU matching; duplicates/ignore/FN; oracle exclusions;
MOT frame/ID conversion; từ chối annotation thưa/identity không chắc/cached floor
không đủ. Đây **không phải thí nghiệm video**. Môi trường thiếu cv2/sklearn/BoxMOT/
YOLO; full DBSCAN integration và inference adapter chưa chạy. Official TrackEval
chưa được khôi phục/chạy, nên chưa xác minh metric evaluator của phiên bản đó.

## Next discriminating experiment — bước nhỏ tiếp theo

1. Cung cấp raw paths + môi trường/lệnh/checkpoint thực tế; xác nhận scene/session,
   calibration và tracker source/YAML. Kiểm kê FPS/resolution/duration rồi xem nhanh
   để chọn 2–3 đoạn 10–20 giây, không coi chúng là các session độc lập.
2. Annotate bbox/ID liên tục trên pilot; giữ visibility và uncertainty. Chỉ thêm
   group references ở interval rõ nếu cần kết luận về grouping.
3. Cache detector một lần ở floor thấp; sweep .10/.15/.20/.25 với model/NMS/imgsz/
   tracker YAML cố định. Đo detection rồi tracking bằng official TrackEval đã kiểm
   tra fixtures. So threshold tốt cho recall với threshold tốt cho association.
4. Replay raw grouping và corrected proximity trên cùng tracks để kiểm tra lỗi
   bookkeeping; tiếp đó so estimated, GT-box→tracker, GT-box+ID và matched-detection
   identity oracle. Oracle matched-only loại unmatched FP/duplicates có báo count;
   không diễn giải như cô lập hoàn toàn association.
5. Nếu chưa có group GT: chỉ inspection downstream. Nếu có group GT đáng tin:
   dùng cùng protocol/frame mask để xem tracking đầu vào hay logic grouping tạo
   degradation. Không cộng các delta thành phần trăm lỗi từng module.

Paper, advisor notes gốc và archived evaluation results được giữ nguyên. Xem
`README.md` cho lệnh baseline/sửa, `ANNOTATION.md` cho format và `AUDIT.md` cho
đường chạy, cấu hình thực có cùng các khoảng trống provenance.
