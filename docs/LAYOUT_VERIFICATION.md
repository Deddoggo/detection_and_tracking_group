# Layout verification — 2026-10-06

Branch: `chore/repository-layout`. No commit, push or history rewrite performed.

- `python -m unittest discover -s tests -v`: **30 tests passed**.
- Parsed **94 Python files** successfully without loading models.
- Four main modules import without ML initialization; root wrappers and canonical direct
  scripts return `--help` successfully even from another working directory.
- Normalized algorithm ASTs match the pre-move working copies for four pipeline stages
  and five shared modules. Normalization excludes imports, path bindings, credential
  retrieval and entry-point wrapping; this is not an end-to-end numerical equivalence test.
- Original GT XML SHA-256 unchanged. Archived evaluation summary, original requirements
  and playground calibration checked byte-for-byte against the pre-move backup.
- All **34 previously tracked media/generated files** removed from the index remain on
  disk, including reference images moved to `assets/reference/`.
- `git diff --check` passed with CRLF-aware whitespace settings.

No detector/tracker/pose inference, new benchmark or GPU validation was run. Main model
dependencies/checkpoints and the server package combination remain unverified. Auxiliary
experiments received syntax/path review, not runtime validation with their external projects.
The root requirements list is not a reproducibility lock.

Pre-existing uncommitted grouping/diagnostic work was retained and included in the pre-move
backup under ignored `.local/`. Source/configuration moves are working-tree changes; the
media untracking is staged. Review both before committing. Never share `.local/`: the backup
also retains original credential-bearing source. The live source reads the key from the
environment; historical Git commits are unchanged and the old key needs rotation before
sharing history.
