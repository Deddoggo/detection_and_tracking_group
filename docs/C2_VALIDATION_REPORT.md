# C2 validation report

2026-11-09. Evidence is consolidated in `c2_validation.json`; full local diagnostics remain under `.local/c2/` and `.local/c1_validation/`.

## Environment actually used

All tests and decoding used:

```powershell
$py = 'C:\Users\trant\miniconda3\envs\paper-research\python.exe'
& $py -B tests/run_c1.py
& $py -B tests/run_c2.py
& $py -B tests/check_active_layout.py
```

`paper-research` exists. Python is 3.13.0; no Python 3.11 environment was created. Native imports and tests establish compatibility for the tested CPU paths.

Latest observed environment: NumPy 2.5.4; OpenCV distribution 4.11.0.86/runtime 4.11.0; sklearn 1.9.1; Transformers 5.22.2; PyTorch 2.15.0+cu134; torchvision 0.29.2+cu134; Pillow 13.0.0; safetensors 0.8.0. Ultralytics 8.4.208 and BoxMOT 26.0.0 were inventoried by package metadata, not inference-tested. Auxiliary `inference` and optional `openpyxl` are absent.

An earlier same-session snapshot reported CPU-only torch 2.15.0. A subsequent snapshot reported CUDA 13.4 and CUDA available. The latter was verified with a one-element CUDA allocation/readback on an RTX 3060 Laptop GPU (6,441,926,656 bytes reported memory). Codex performed no package installation, removal or update. Both snapshots are recorded; no cause for the external environment change is asserted. This scalar operation is not model validation.

## Tests: expected versus observed

| Check | Expected | Actual / status |
|---|---|---|
| Active source/resource checks | Compile without models; approved helpers/resources present | 3 PASS. |
| C1 regressions | Preserve B0/C1 semantics with explicit compatibility mode | 56 PASS, 1 SKIP among 57 methods. Native sklearn chain/split tests executed. |
| Excel export | Native pandas/openpyxl when both available | SKIP: openpyxl absent. No substitute writer or installation. |
| C2 XML | Zero-based IDs, float boxes, flags and raw partial labels preserved | PASS, including actual local XML coverage and label counts. |
| Track 40 | Outside at 319; absent 320–333; returns at 334 | PASS; parser never interpolates this interval. |
| Source timestamps | Decode and packet PTS agree throughout `[0,1800)` | PASS: 1,800 frames, maximum error 1.4210854715202004e-14 seconds. |
| Wrong decoded PTS | Reject mismatched timestamp | PASS on a deliberately controlled decoder response. |
| Schema round-trip | Preserve raw arrays/scores and references exactly | PASS, including scores above 1 without clipping. |
| Missing/excluded/empty | Retain exclusions and honest missingness; allow empty frames | PASS. |
| Invalid/tampered/incomplete cache | Reject bad shapes, NaN/Inf, duplicate IDs, ordering/count/hash/source errors | PASS. |
| Metric provenance | Reject missing/unverified/wrong-video/wrong-H/incorrect correspondence evidence | PASS on synthetic contracts. Actual field validity NOT VERIFIED. |
| Pixel diagnostics | Require explicit pixel close-distance and no H | PASS. |
| Native grouping | Preserve chain merging and one-to-one split IDs | PASS with sklearn, no replacement clustering. |
| Direct raw observations versus loaded cache | Identical replay predictions under identical recipe/state | PASS on explicitly synthetic poses. |
| Retained live analyzer versus replay | Same EMA, projected positions and group IDs | PASS with injected controlled poses, without constructing neural models. |
| Neural-free CLI replay | No torch/Transformers/inference/Ultralytics/BoxMOT import | PASS: subprocess import guard rejects those modules. |
| Replay determinism/state | Identical repeats; correct reset after empty sample; reject skipped/reordered initial samples | PASS; input observations unchanged. |
| Missing checkpoint | Fail without downloading/substituting | PASS on empty local directory. |
| Real ViTPose smoke/cache | Use supplied identified weights | NOT EXECUTED: weights unavailable. |

C2 suite: **25 PASS**, zero failures/skips with the local XML present. Together with C1/layout: **84 PASS / 1 SKIP**. These are test methods, not independent benchmark sequences.

Initial validation exposed a C1 test-harness issue: `patch.dict(sys.modules)` removed newly imported OpenCV Python modules while retaining native extension state. Preloading OpenCV before substitution fixed the tests; no environment package was changed. An initial direct C2 run also hit Windows sandbox temporary-directory permissions. Its approved outside-sandbox retry passed; the final runner uses an isolated ignored workspace temporary directory and passes inside the sandbox.

## Real data evidence

Source: `WIN_20260303_10_09_56_PRO_GT_First60s/WIN_20260303_10_09_56_Pro.mp4`.

- Video SHA256: `fa1044aa1f3c9d413bdd608d39f8545c593b34457f5ecb6dc80a28beb62cb119`.
- XML SHA256: `67addc127c07d98c17d0c160dc03d5f4025e5926dfb4d2d786aa901af37f7f08`.
- 1920x1080, 2,956 frames; time base 1/30000; variable PTS increments. No fixed-30-FPS assumption.
- XML: CVAT 1.1 job 4549037, 76 persons, declared frames 0–2955.
- First 1,800 frames: 71,157 visible person boxes, including 66,223 `unreviewed` and 4,934 assigned G1–G9 attribute values. These remain partial, unfinished reference annotations.
- Source frame 1799 is at 65.18396666666666 seconds. `[0,1800)` is a frame interval, not literal 60 seconds.
- Decoding verified all first 1,800 frame positions/PTS. Overlays were generated for 0,1,3,30,300,319,334,899,900,1799. Frames 0,334,1799 were visually inspected: broad person-box correspondence and no obvious global offset/resizing error. Individual box precision, identity continuity and group correctness are not certified.

Local timing/review evidence: `.local/c2/alignment_20261109/alignment.json` and the separately created `alignment.reviewed.json`. The latter retains all source PTS and records the limited visual review. Neither changes the XML/video.

Reproduce alignment into a **new** output directory:

```powershell
& $py -B analyze_subgroups.py validate-data `
  --video 'WIN_20260303_10_09_56_PRO_GT_First60s/WIN_20260303_10_09_56_Pro.mp4' `
  --xml 'WIN_20260303_10_09_56_PRO_GT_First60s/annotations.xml' `
  --end 1800 --output '.local/c2/alignment_NEW_RUN'
```

Review newly generated overlays and record a separate reviewed alignment file before real cache export. No full-video pose/grouping inference or public benchmark was run.
