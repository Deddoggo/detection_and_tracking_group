# Audit pipeline thực tế — 06/10/2026

Phạm vi: `detect_yolo.py → track.py → stitch_tracks.py → analyze_subgroups.py`,
cùng `identify_group_social.py`, `orientation_utils.py` và dependencies trực tiếp.
Audit working tree hiện tại trên HEAD `2db7aac`, gồm các sửa correctness từ phiên
trước; không đồng nhất với code đang có trên server hoặc archived thesis run.
Tái sử dụng findings/tests đã có; phiên này chỉ đọc code và viết ghi chú, không
chạy inference/test mới, không đổi thuật toán. Theo phạm vi bạn cung cấp, GT
bbox/person ID 0–900 cho phép đánh giá detection/tracking; chưa có group GT.

**Kết luận:** giao diện bbox/frame cơ bản nhất quán. Rủi ro lớn về implementation
là input không được ràng buộc cùng nguồn, calibration fallback đổi đơn vị âm thầm,
và orientation gate mặc định không chọn lọc. Stitching và group continuation có
các giả định cần đo; chưa thể chọn bottleneck chính chỉ bằng source.

## Findings

**1. Đúng format, nhưng thiếu kiểm tra tính toàn vẹn giữa các bước — xác nhận code.**
[detect_yolo.py:69](../pipeline/detect_yolo.py),
[track.py:61](../pipeline/track.py),
[stitch_tracks.py:181](../pipeline/stitch_tracks.py),
[analyze_subgroups.py:135](../pipeline/analyze_subgroups.py).
JSON dùng pixel `xywh`, key `frame_00000` zero-based; tracker nhận `xyxy`, pose
được đổi lại COCO `xywh` đúng thứ tự. Stitching giữ bbox/frame và đổi ID. Tuy nhiên
không kiểm tra video identity/range; `.get(frame_key, [])` biến frame thiếu thành
“không có người”. Analyzer ưu tiên bất kỳ file stitched còn tồn tại, chỉ in tên
file chứ không xác minh cùng run. Cache cũ/sai video có thể chạy tiếp với kết quả
hợp lệ về cú pháp nhưng sai nội dung. Bbox bị ép integer tại detection và tracking:
mất độ chính xác dưới pixel, ảnh hưởng tương đối lớn hơn với người nhỏ; chưa đo.

**2. Detection cutoff cắt mất bằng chứng trước association — xác nhận; tác động chưa đo.**
[detect_yolo.py:22](../pipeline/detect_yolo.py),
[detect_yolo.py:53](../pipeline/detect_yolo.py),
[track.py:70](../pipeline/track.py).
Detector đặt conf=.15, IoU=.45, imgsz=1280; tracker chỉ nhận conf>.15. Bbox thấp hơn
ngưỡng đã mất khỏi cache và không thể được low-confidence association phục hồi.
Không đồng nghĩa ByteTrack mất toàn bộ low-score stage: còn phụ thuộc các ngưỡng
trong version thực tế. Không có duplicate suppression bổ sung trong scripts;
hiệu lực của tham số IoU/NMS và duplicate removal bên trong model/tracker cần đối
chiếu implementation cài trên server. Không suy ra rằng giảm conf chắc chắn tốt hơn.

**3. Hiệu lực tracker overrides chưa xác minh — nghi ngờ có căn cứ.**
[track.py:24](../pipeline/track.py).
`track_thresh=.25`, `track_buffer=90` được gán **sau** constructor dùng YAML trong
package. Nếu constructor đã tính `max_time_lost` hoặc dùng tên threshold khác,
hai assignment có thể không thay đổi behavior mong muốn. Source/YAML BoxMOT của
server không có trong repo: chưa kết luận association, confirmation, lost-track
persistence/reactivation hoặc duplicate removal bị lỗi. Không cần viết lại các
chức năng đó trước khi biết chúng đang hoạt động thế nào.

**4. Stitching có temporal non-overlap guard, nhưng appearance/motion evidence yếu — cơ chế xác nhận, false joins chưa đo.**
[stitch_tracks.py:95](../pipeline/stitch_tracks.py),
[stitch_tracks.py:123](../pipeline/stitch_tracks.py).
Ghép greedy theo histogram HSV trung bình toàn tracklet, gap≤150 frame,
distance≤350 pixel, similarity>.60; không xét velocity, uncertainty hoặc chênh lệch
best/second-best candidate. Hai trẻ cùng màu áo/crossing dễ trở thành ứng viên
tương tự. Guard `start_f <= end_f` ngăn ghép hai tracklet chồng thời gian; không nên
quy cho code này một lỗi tạo simultaneous duplicate ID chưa có bằng chứng.
Sau nối, root cập nhật end/last_box nhưng **không cập nhật avg_hists[root]**
([stitch_tracks.py:164](../pipeline/stitch_tracks.py)):
link tiếp dùng vị trí segment mới với appearance segment gốc. Đây là trạng thái
không đồng bộ cần chủ ý xác định policy. Histogram toàn tracklet sử dụng tương lai,
nên kết quả là offline refinement. Link sai sẽ truyền cùng person ID qua hai
người, làm sai membership overlap và lịch sử orientation downstream.

**5. Frame-based timing không phải thời gian thực — xác nhận.**
[stitch_tracks.py:14](../pipeline/stitch_tracks.py),
[analyze_subgroups.py:243](../pipeline/analyze_subgroups.py).
150 frame chỉ là5 giây nếu đúng30FPS; với VFR không có một quy đổi cố định.
Analyzer xử lý mỗi3 frame: khoảng0–900 chỉ có301 lần cập nhật, vẫn giữ source
index đúng nhưng không có records cho frame bị skip. Group events/pose changes
giữa các mẫu bị bỏ qua. Đây không phải off-by-one; sai diễn giải xảy ra nếu coi
output là dense hoặc dùng frame count làm duration.

**6. Tiny-box filter loại người khỏi cả subgroup, không chỉ pose — xác nhận.**
[analyze_subgroups.py:153](../pipeline/analyze_subgroups.py).
`w<15 or h<30` xảy ra trước khi thêm bbox/ID vào grouping input. Người xa có thể
biến mất khỏi spatial clusters dù tracking vẫn có ID; dao động qua ngưỡng gây
membership bật/tắt. Chưa đo bao nhiêu GT person bị loại. Tách pose eligibility
khỏi person existence là engineering policy cần sửa trước khi quy mọi miss cho
detector; cách giữ/loại track chất lượng thấp vẫn cần kiểm chứng.

**7. Pose failure/staleness bị che khuất trong output — xác nhận.**
[analyze_subgroups.py:70](../pipeline/analyze_subgroups.py),
[analyze_subgroups.py:166](../pipeline/analyze_subgroups.py),
[analyze_subgroups.py:199](../pipeline/analyze_subgroups.py),
[analyze_subgroups.py:257](../pipeline/analyze_subgroups.py).
Lỗi predict trả `None` không ghi nguyên nhân. Khi ID còn hiện diện, raw direction
thiếu sẽ dùng hướng cũ vô thời hạn; chỉ một sampled frame vắng ID thì history bị
xóa. ID switch liên tục có thể trộn lịch sử hai người. EMA ưu tiên raw hiện tại
.7, không dựa quality/elapsed time. JSON cuối không giữ keypoints, direction,
confidence, age hay failure reason; khó phân biệt pose fail, stale cue và grouping
logic. Hiện legacy gate không dùng chúng để loại cặp; rủi ro membership từ stale
cue trở nên trực tiếp nếu chuyển sang gate có tác dụng.

**8. Orientation gate hiện tại thực chất là proximity-only — đã xác nhận trước, vẫn áp dụng.**
[analyze_subgroups.py:220](../pipeline/analyze_subgroups.py),
[identify_group_social.py:29](../modules/identify_group_social.py),
[orientation_gate.py:24](../modules/orientation_gate.py).
Caller không truyền mode nên dùng `legacy`. Gate OR trên absolute cosines luôn
accept vector2D đơn vị hợp lệ: nếu cả hai facing magnitudes≤.3 thì mutual absolute
cosine≥.82>.7. Cue thiếu cũng accept; dist<.5 bypass. Do đó orientation không
prune proximity edges trong đường chạy này. Chỉ đổi OR thành AND không xác lập
social compatibility. `facing_experimental` đã tồn tại nhưng caller không truyền
pose_confidences; chỉ bật tên mode vẫn fallback accept. Bản sửa numeric group IDs
không thay đổi finding này.

**9. Image orientation và geometry chưa đủ cơ sở để gọi metric ground heading — xác nhận thiết kế; sai số chưa đo.**
[orientation_utils.py:34](../modules/orientation_utils.py),
[orientation_utils.py:75](../modules/orientation_utils.py),
[identify_group_social.py:47](../modules/identify_group_social.py),
[analyze_subgroups.py:129](../pipeline/analyze_subgroups.py).
Hướng là shoulder normal + ear→nose, weights1/1.5 rồi ép Y×.5; joint confidence
không chứng minh đúng heading/sign. Chiếu vector bằng footpoint và endpoint cách
10 pixel chỉ hợp lý cho image displacement tương ứng ground tangent; không tự
biến projected body/head cue thành heading đúng. Bbox bottom-center là footpoint
proxy, lệch khi occlusion/jumping/truncation. Calibration JSON chỉ chứa matrix,
không source points/measurement provenance. **Nếu file thiếu**, code chuyển sang
pixel coordinates nhưng vẫn dùng eps=1.75 và close gate=.5: lỗi đổi đơn vị âm
thầm, thường làm gần như mọi người isolated. File hiện có nên không khẳng định
fallback đang xảy ra hoặc matrix hiện tại sai.

**10. Distance connectivity có thể nối chuỗi thành một nhóm lớn — xác nhận behavior, chưa phải bug.**
[identify_group_social.py:87](../modules/identify_group_social.py),
[identify_group_social.py:122](../modules/identify_group_social.py).
Với min_samples=2 và gate hiện tại, DBSCAN cho các connected components có ít
nhất2 người trong graph khoảng cách≤eps. A gầnB, B gầnC đủ ghépABC dù A–C>eps.
Một FP/duplicate hoặc người đi ngang có thể làm cầu nối hai cụm; miss ở cầu nối
có thể gây split. Đây là spatial clustering, chưa là social-group membership.
Isolated people dùng chung sentinel id_g=-1, không được gom tất cả thành một nhóm.

**11. Group-ID overwrite đã sửa; continuation chưa đồng nghĩa membership ổn định.**
[identify_group_social.py:131](../modules/identify_group_social.py),
[group_identity.py:24](../modules/group_identity.py).
Working tree hiện tại dùng one-to-one assignment, không còn nhiều cluster overwrite
cùng old ID như bản archived. Retention là intersection/old_size≥.5; một nhóm2
người còn1 người giữa nhiều người mới vẫn có thể giữ numeric ID. Matching greedy,
split giữ ID cho strongest eligible child, merge lấy một parent; tie deterministic.
Không có persistence qua frame không cluster/noise; một lần miss có thể kết thúc
group ID. Split/merge lineage được tính nhưng `_events` bị bỏ, không lưu trong
JSON cuối. Person-ID switches/false stitching trực tiếp đổi overlap, có thể tạo
false continuation/split/merge. Sửa bookkeeping là correctness, không phải bằng
chứng cải thiện social membership.

## Ưu tiên cải tiến

1. **Correctness trước:** ràng buộc source/cache/frame completeness; từ chối
   calibration thiếu/sai đơn vị; xác minh effective tracker overrides; ghi pose
   failure/age và split/merge evidence. Giữ group-ID fix hiện có. Không thêm thuật
   toán association mới để “sửa” dependency chưa audit được.
2. **Đo detection/tracking:** phân biệt loss do detection, ByteTrack và stitching
   bằng cùng GT0–900. Kiểm tra tiny-person recall, duplicates, IDSW/Frag, IDF1 và
   AssA trước/sau stitching. Chưa chọn threshold, NMS hay persistence là cải tiến
   đầu tiên cho tới khi biết lỗi chi phối.
3. **Method candidates có điều kiện:** nếu false joins chi phối, xem xét stitching
   có ambiguity rejection + motion/time evidence; nếu low-score true detections
   đang bị cắt, kiểm tra operating point phù hợp association; nếu subgroup dễ
   bridge/flicker, xem xét robust grouping với input uncertainty. Orientation chỉ
   nên là candidate sau khi xác thực cue/sign/reliability; membership improvement
   cần group reference riêng, không được suy từ arrows hoặc numeric ID ổn định.

## Câu hỏi khi có output server

- Detector bỏ sót bao nhiêu GT theo bbox size/occlusion, còn bao nhiêu duplicate?
  Lọc15×30 trong analyzer loại thêm những GT nào?
- BoxMOT thực tế đọc ngưỡng nào; persistence/confirmation/reactivation đang có
  hiệu lực ra sao? Có low-score true detections hữu ích nhưng bị cutoff loại trước?
- Stitching tăng IDF1/AssA hay chỉ giảm số IDs/Frag? Các link sai có tập trung vào
  áo tương tự, crossing, long gap hoặc chain dùng histogram root cũ không?
- File stitched, video và calibration có cùng source/run không; footpoint
  projection có hợp lý ở vùng xa/occlusion? Đây là geometry evidence, chưa là
  kiểm định metric accuracy.
- Tỷ lệ pose missing/stale và tuổi cue là bao nhiêu? Proximity versus legacy có
  đúng cùng edges/members như code dự đoán? Có chain bridges hoặc group resets
  trùng với person-ID errors không? Chưa có group GT nên chỉ kết luận correctness
  và qualitative failure cases, không công bố grouping accuracy.
