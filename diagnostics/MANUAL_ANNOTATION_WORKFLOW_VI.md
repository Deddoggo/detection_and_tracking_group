# Gán nhãn thủ công pilot01 — bbox + person ID

## Gói đã chuẩn bị

Thư mục `diagnostics/manual_annotation_pilot01/` chứa clip `pilot01_30-45s.mp4`,
`frame_mapping.json`, `frame_mapping.csv`, `source.json`, cấu hình nhãn CVAT và
template group reference riêng. Chỉ dùng clip đã có mapping với
`all_decoded_pixel_hashes_match_source: true`; không dùng các file `.unverified.mp4`.

Đoạn gốc khoảng 30–45 giây: **source frame832 đến1250**, zero-based, tổng419 frame.
Clip có frame0–418. Frame clip i tương ứng source frame832+i, MOT frame i+1.
Giữ nguyên1920×1080. Video gốc VFR; clip dùng đồng hồ trình chiếu đều để CVAT đọc
ổn định, dài15 giây. Không mất/lặp frame và không thay pixel đã decode; mapping
lưu timestamp gốc từng frame. Khi phân tích chuyển động hãy dùng timestamp GỐC,
không nhân frame với60 hay coi timestamp trình chiếu là timestamp gốc.

Đây là một pilot trong một video, không phải benchmark độc lập. Chưa có GT thật:
các template còn pending. Không chạy detector/tracker/pose hoặc tự tạo pseudo-label.

## 1. Công cụ: CVAT local

Bạn xác nhận chưa có công cụ annotation. Chọn **CVAT Community chạy local**, vì
Track mode hỗ trợ rectangles, persistent tracks và interpolation; native XML/ZIP
export giữ tracks/attributes. Tham khảo [Track mode](https://docs.cvat.ai/docs/annotation/manual-annotation/modes/track-mode-basics/)
và [định dạng CVAT](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/).
Docker đã có executable nhưng engine hiện chưa chạy. Chưa cài/chạy CVAT trong phiên
này; chỉ thiết lập khi máy có điều kiện. Không cần GPU cho annotation thủ công.

Để thiết lập local, mở Docker Desktop, chờ engine sẵn sàng rồi làm theo
[hướng dẫn cài chính thức](https://docs.cvat.ai/docs/administration/community/basics/installation/).
Workflow cơ bản trong PowerShell, chạy ở thư mục bạn muốn lưu công cụ:

```powershell
docker info
git clone https://github.com/cvat-ai/cvat.git cvat-local
Set-Location cvat-local
git rev-parse HEAD
docker compose up -d
docker compose exec cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Lưu commit CVAT và phiên bản hiển thị của ứng dụng vào ghi chú annotation; các
lệnh setup trên chưa được chạy/kiểm chứng local vì Docker engine chưa hoạt động.
Nếu bản bạn cài có hướng dẫn khác, theo hướng dẫn chính thức của phiên bản đó.
Sau khi server sẵn sàng, mở `http://localhost:8080`, đăng nhập bằng tài khoản local
bạn vừa tạo. Không cần cấu hình AI/serverless/auto-annotation. Gói clip/converter
không phụ thuộc vào dịch vụ đã chạy, nên bạn có thể hoàn tất setup sau.

## 2. Tạo đúng một task cho clip

1. Đặt tên task `playground_pilot01_persons`. Upload **chỉ**
   `diagnostics/manual_annotation_pilot01/pilot01_30-45s.mp4`.
2. Label duy nhất: `person`. Tạo các attributes dưới đây; JSON mẫu trong
   `cvat_labels.json` giúp nhập ở raw labels editor nếu phiên bản có hỗ trợ.
3. Frame step=1; không skip/delete frame, không flip/resize/crop. Một job trọn clip
   là đủ; segment size419 trở lên. Không chia clip thành nhiều jobs trong pilot này.
4. Sau import, kiểm tra task có419 frames, frame0–418, ảnh1920×1080. So frame đầu/
   cuối với `alignment_first.png`, `alignment_last.png` để tránh nhầm clip.

| Attribute | Kiểu | Mutable | Default | Cách dùng |
|---|---|---|---|---|
| person_id | text | false | rỗng | nhập số nguyên dương1,2,3… riêng cho mỗi người |
| identity_uncertain | checkbox | true | false | true ở các frame không xác định được identity |
| occlusion | select | true | none | none / partial / heavy |
| truncated | checkbox | true | false | người bị cắt bởi biên ảnh |

Không cần orientation, gender hoặc group attribute trong task này. `person_id` là
ID reference của người gán nhãn; không nhầm với số track mà UI hiển thị. ID custom
giúp lớp group sau này ổn định qua các lần export. Không tái sử dụng ID cho người khác.

## 3. Quy tắc bbox, identity và che khuất

- Label **mọi người discernible**, cả trẻ và người lớn, kể cả nhỏ/ở xa/biên ảnh.
  Không gọi người lớn là FP của detector person. Nếu muốn chỉ nghiên cứu trẻ sau
  này, định nghĩa scope riêng; chưa suy đoán tuổi/gender ở bước này.
- Rectangle bao sát **phần người đang nhìn thấy**, không bao bóng, cây, xe, nền,
  và không vẽ thêm chân/thân bị che. Nếu nhiều phần của cùng người nhìn thấy,
  bbox bao các phần đó. Người truncated: bbox dừng ở biên ảnh, bật truncated.
- Một người có đúng một bbox/frame và một `person_id` liên tục. Quan sát quần áo,
  vị trí và chuyển động bằng mắt; không dùng model output để quyết định identity.
- Người bị che một phần: vẫn bbox phần nhìn thấy; bật CVAT **Occluded** và đặt
  `occlusion=partial/heavy` tương ứng. Binary Occluded phải khớp custom category.
- Người biến mất hoàn toàn hoặc ra khỏi ảnh: dùng **Outside** từ frame đầu tiên
  không nhìn thấy. Không để interpolation tạo bbox cho người hoàn toàn vô hình.
- Reappearance: nối lại cùng track/ID **chỉ khi chắc là cùng người**. Nếu không
  chắc, tạo một đoạn track/ID mới và đánh dấu identity_uncertain ở đoạn cần review;
  ghi lý do vào notes. Converter giữ bbox nhưng đặt person_id=null tại frame
  uncertain; không đoán ID để giúp export tracking pass.

## 4. Gán nhãn theo Track mode, từng bước nhỏ

1. Xem clip một lượt ở tốc độ chậm để hiểu chuyển động. Chưa vẽ trong lượt này.
2. Tại frame0 chọn Rectangle, label person, **Track**, rồi vẽ một người dễ nhìn.
   Điền person_id=1. Di chuyển5–10 frames, chỉnh bbox; thao tác chỉnh tạo keyframe
   và CVAT nội suy các frame giữa. Đây là interpolation từ nhãn thủ công, không AI.
3. Rà các frame giữa hai keyframes. Khi chạy nhanh/crossing/đổi hướng, đặt keyframes
   dày hơn, có thể từng frame. Đừng mặc định bbox nội suy đã đúng.
4. Làm hết quãng xuất hiện của người đó, Outside/reappearance đúng frame, rồi tiếp
   người tiếp theo. Với người đứng yên vẫn kiểm tra họ có di chuyển trong15 giây.
5. Người đi vào sau frame0: bắt đầu track ở frame xuất hiện đầu tiên; không kéo bbox
   ngược về đầu clip. Save thường xuyên. Cuối cùng rà lại **toàn bộ419 frames**,
   đặc biệt vùng bóng tối/cây bên trái, người nhỏ, foreground/biên ảnh và overlap.

Bắt đầu bằng1–2 người dễ trước để học thao tác. Có thể save bản dở; chỉ xác nhận
full coverage sau khi đã kiểm tra tất cả người và interpolation trong cả clip.
Khối lượng không yêu cầu mở rộng sang video106 giây hoặc toàn bộ74 phút.

## 5. Export, chuyển đổi và xem lại

Sau khi save, **Export annotations của task**:

- Giữ bản **CVAT for video1.1** để backup tracks/keyframes.
- Ưu tiên bản **CVAT for images1.1**, không Save images, cho converter: CVAT đã
  giải interpolation thành bbox từng frame và giữ `track_id`. Vẫn gán nhãn bằng
  video Track mode; đây chỉ là format export. Converter cũng hỗ trợ native video
  rectangle keyframes, nhưng dense export giúp tránh lệch interpolation giữa bản.

Lưu export ZIP dưới `diagnostics/manual_annotation_pilot01/exports/persons_v1.zip`.
Không đổi extension ZIP thành XML. Script đọc ZIP trực tiếp, không chạy model.
Mỗi lần sửa/export dùng version và thư mục output mới để giữ lịch sử.

Chạy từ root repository. Converter/check chỉ dùng Python standard library; lệnh
preview dùng Pillow đã có trong Python bundled dưới đây:

```powershell
$annPython = 'python'
$annRoot = 'diagnostics/manual_annotation_pilot01'
& $annPython -m diagnostics.cvat_annotations convert --export "$annRoot/exports/persons_v1.zip" --mapping "$annRoot/frame_mapping.json" --out-dir "$annRoot/converted_v1_pending"
& $annPython -m diagnostics.cvat_annotations check --input "$annRoot/converted_v1_pending/annotations.json"
& $annPython -m diagnostics.cvat_annotations preview --input "$annRoot/converted_v1_pending/annotations.json" --mapping "$annRoot/frame_mapping.json" --frames 0 100 209 300 418 --out-dir "$annRoot/preview_v1" --ffmpeg 'ffmpeg'
```

Xem PNG overlay trong preview_v1 và rà lại trong CVAT nếu bbox/ID lệch. Script
kiểm tra size/range419, bbox finite nằm trong ảnh, IDs không trùng trong frame,
Outside/uncertainty, frame coverage và chuyển source indices. Nó không thể phát
hiện người bạn chưa vẽ hoặc chứng minh identity đúng chỉ bằng cấu trúc file.
Danh sách empty_frames_to_review và warnings cần xem lại, không tự bỏ qua.

## 6. Xác nhận review rồi tạo annotation evaluator

Khi đã rà hết clip, tạo checklist gắn hash vào đúng mapping và đúng export:

```powershell
& $annPython -m diagnostics.cvat_annotations make-review --export "$annRoot/exports/persons_v1.zip" --mapping "$annRoot/frame_mapping.json" --out "$annRoot/review_v1.json"
```

Mở review_v1.json, điền reviewer/review_date và chỉ chuyển sáu ô false→true nếu
thực sự đã hoàn thành. Không sửa hashes. Sau đó:

```powershell
& $annPython -m diagnostics.cvat_annotations convert --export "$annRoot/exports/persons_v1.zip" --mapping "$annRoot/frame_mapping.json" --review "$annRoot/review_v1.json" --out-dir "$annRoot/converted_v1_reviewed"
& $annPython -m diagnostics.cvat_annotations check --input "$annRoot/converted_v1_reviewed/annotations.json"
```

`annotations.json` dùng đúng schema evaluator hiện có: absolute source frame,
pixel xywh, person ID, full coverage sau human review; giữ timestamps/uncertainty
và metadata export. Nếu còn uncertain identity thì detection reference vẫn hữu ích,
nhưng strict full-clip tracking export bị chặn. Không xóa các người khó để vượt kiểm
tra; giải quyết uncertainty hoặc sau này chọn một interval liên tục đáng tin có
ghi exclusions. File dở chưa review không được gọi là GT hoàn chỉnh.

Khi qa_report ghi tracking_export_ready=true và warnings rỗng, có thể export MOT
GT mà **không chạy evaluation/inference**:

```powershell
& $annPython -m diagnostics.cli mot-export --input "$annRoot/converted_v1_reviewed/annotations.json" --out "$annRoot/gt_v1.txt"
```

MOT dùng frame1–419; metadata bên cạnh giữ mapping IDs và source. Bbox convention
visible extent cần được giữ thống nhất khi so detector/tracker sau này. Visibility
fraction không được tự suy từ binary occlusion; nếu chưa gán, export ghi sentinel
unknown và không sử dụng nó để lọc GT. Chưa tính HOTA/IDF1 hay bất kỳ metric nào.

## 7. Group membership là lớp riêng, làm sau

`groups_pending.json` chỉ là template, không có membership labels. Sau khi cố định
person GT v1, tham chiếu hash bản đó và custom person_id. Dùng intervals theo frame
gốc, định nghĩa nhóm quan sát được cùng thầy/LAPSCO, explicit singleton và unknown.
Không dùng CVAT shape grouping, DBSCAN proximity hoặc track ID continuity làm social
group GT. Chưa cần điền lớp này để hoàn thành pilot bbox+ID.
