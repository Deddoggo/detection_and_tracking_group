# Repository cleanup and branch history

## Preserved state and safe workflow

Original master is `2db7aac2fca6f03e1a974b1c46a921ab54330160`, archived exactly as
`archive/pre-cleanup-master`. Clean master is a new descendant of that original
commit, not a descendant of either correctness branch. All nine active source
files and historical `requirements.txt` remain unchanged in the cleanup commit.

The initial checkout was `chore/repository-layout` at
`8174be6397f38a8a7713e2d053e663f0baf93292`, one commit ahead of its remote-tracking
branch. That branch remains intact; its directory migration/candidate pipeline
was not used as the new baseline. No tracked user edits were pending.

Earlier C1 work is preserved before rebuilding:

| Reference | Preserved tip |
|---|---|
| `archive/c1-active-before-cleanup-2026-11-09` | `697f96eef71f6649ae8d11c07432b21c0010eb15` |
| `archive/c1-broad-before-cleanup-2026-11-09` | `2004408158e9db7a70998a85fa0ce5c27bc81037` |

The existing active-C1 branch was renamed to its archive ref before its name was
reused from cleaned master. No reset, clean, force checkout, cherry-pick of the
over-scoped commit, or published history rewrite was used. The redundant local
`fix/c1-correctness-and-tests` is eligible for deletion only after the rebuilt C1
commit and focused checks succeed; its tip remains reachable via the archive ref.

## Dependency-based cleanup

Inspection covered imports, runtime path literals, package configuration lookup
and script entry points in all nine authoritative files. The active local imports
are `analyze_subgroups` -> `identify_group_social` and `orientation_utils`.
BoxMOT's ByteTrack config is resolved within its installed package. None of the
retired detector/tracker/lifting scripts or old local tracker YAMLs is referenced
by the active path. No subprocess invocation of retired scripts was found.

The cleanup removes 24 unused Python/config files from the tracked working tree:
legacy detection/tracking/evaluation scripts, old integrated YOLO demonstrations,
MixSTE/lifting/3D-render experiments, interactive legacy measurement/calibration
tools and the unused BoostTrack/DeepOCSORT YAMLs. Their complete paths are in
`repository_state.json`; contents/history remain available on the original archive.

Thirty-four historical artifact files are untracked **with `git rm --cached`**,
not deleted from disk: videos, pose pickles, generated images/GIFs, the workbook,
the archived metric log and MixSTE run log. Ignore rules retain these locally
without adding videos, weights, private annotations or outputs to new commits.
The calibration reference images and all five existing calibration matrices are
retained as source resources; no parameters or correspondences were changed.

Clean master contains:

- Nine flat active/optional Python sources, without C1 helper modules or fixes.
- `Real_playground.png`, `Real_playground_2.png`, five calibration JSONs.
- README, user-provided AGENTS, Git configuration, unchanged historical
  requirements and a direct-import `requirements-active.in` inventory.
- Three stdlib source/layout/resource checks and this cleanup report/JSON manifest.

`requirements-active.in` is a declaration inventory, not a validated version lock;
no dependency upgrades/installs or models were downloaded. Missing runtime data,
package/version compatibility and physical calibration remain unresolved.

## Rebuilt C1 scope

The rebuilt `fix/c1-active-pipeline-only` starts at the cleanup commit. Transfer
is selective: three corrected active files, four used helpers, and the narrowed
regression suite. Legacy grouper/tracker/MixSTE changes are excluded. Existing
C1 engineering logic is transferred without selecting a new scientific method.
Prior full reports remain on the preserved refs; they are not copied into the
production tree. B0 source comparisons load only the original active grouper
from its Git object, avoiding a duplicate baseline source tree.

| Changed source | Purpose |
|---|---|
| `identify_group_social.py` | Split-safe one-to-one IDs, monotonic allocator, explicit empty active state, contracts, B0 diagnostic dispatch |
| `orientation_utils.py` | Finite/shape contracts and nonmutating direction postprocess |
| `generate_report.py` | Explicit source timing, safe output, versioned descriptive report schema |
| `group_identity.py` | Pure deterministic asymmetric old-coverage association |
| `orientation_gate.py` | Preserved B0 and explicitly separate diagnostic predicates |
| `grouping_contracts.py` | Used bbox/ID/direction/calibration/projection contracts |
| `report_metrics.py` | Defined group/session/time/area/Unknown semantics |
| `tests/c1_cases.py`, `tests/run_c1.py` | Active-only regression tests and explicit native/substitute evidence |

Models, original calibration values, DBSCAN and the scientific default predicate
remain unchanged. Group-ID bookkeeping and new report schema are explicitly
different from B0. C1 does not fix full-pipeline missing-H semantics or validate
social-group accuracy. C2/C3/C4/H2/H3/H4 were not started.

## Preservation and validation

Before switching branches, all refs/remotes/upstreams and 94 existing
untracked/ignored files were inventoried and hashed locally. User AGENTS, video,
GT export and the untracked audit were preserved. Local inventories and recovery
evidence stay under `.local/repository_cleanup_2026-11-09/`; private file inventories
are not committed. Each of the 34 untracked historical artifacts also had its
content hashed before untracking. Post-cleanup hashes are checked again.

Clean-master checks: `python -B tests/check_active_layout.py` passes three tests;
all nine sources compile without execution, required local imports/resources are
present, and no retired root scripts remain. `git diff` against the original
archive proves no active code or historical dependency change during cleanup.
Corrected C1 checks and their native-dependency limits are recorded separately
in `C1_VALIDATION.md` and `c1_validation.json` on that branch. No model imports,
GPU inference, dataset processing or benchmark was run for cleanup.

## Remote status

Read-only GitHub checks identified authenticated account `Deddoggo` and verified
that origin is `Deddoggo/detection_and_tracking_group`, a fork of
`vbarra/detection_and_tracking_group`, with push permission. The requested archive
push was declined, so **no remote branches were changed**. No upstream remote,
remote deletion, default-branch change, permissions change or force-push was used.

At inspection `origin/master` pointed to the original `2db7aac`, so the local clean
master is designed as a normal fast-forward. Before any later user-approved
push, recheck remote heads; if remote master has diverged, leave it unchanged.
The local Git master and GitHub's default-branch setting are distinct.
