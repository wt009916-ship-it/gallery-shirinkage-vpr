# Reproducibility entry

This release separates immediate aggregate reanalysis from image-level reruns.
Neither a successful checksum nor a regenerated table certifies novelty, model
training reproduction, or a risk guarantee.

## 1. Clean aggregate reproduction (Python 3.10+, standard library)

From the public repository root:

```powershell
python -I verify_artifacts.py
python -I experiments/regenerate_paired_table.py --output-dir derived
```

The second command reads only `results/paired_evidence/normalized.csv` and
`paired_summary.json`. It reconstructs paired feasible case sets and recomputes
the primary means, producing `derived/paired_primary.csv` and
`derived/reproduction_check.json`. It retains unavailable cases in denominators,
checks zero-answer risk is absent, and keeps model, cohort, region, budget and
label radius separate. It does not need the original workspace, PyTorch, private
manuscripts, GPU, raw images, credentials or network access.

The normalized file contains 75,866 metric rows, not independent observations.
The 2,046 summary groups include all previously reported answer-budget and
empirical-threshold regimes. Fixed-budget comparisons have equal answer counts;
threshold comparisons may not. No new thresholds are fitted here. The legacy
University appearance/threshold analysis is kept in a separate file because its
policy definition differs. Repeated gallery masks are not independent sites.

## 2. New deployment evidence

`results/history_access/summary.json` reports three modes: active gallery only,
same-query cache lookup, and old-gallery recomputation without query history.
Archive storage and query cache storage are different costs. New queries require
archived descriptors; a small cached query state cannot replace them. Timing is
warm local single-query FP32 scoring and stable full sorting. End-to-end image
measurements include decoding, file verification, encoding and feature transfer,
but exclude model loading and risk calibration. The 32-query-per-flight slice
is an operational history-free replay of already evaluated images.

`results/history_selector/` contains a single-rule exploratory reanalysis on
DenseUAV. Both regions had been examined earlier; this is not fresh validation.
The rule failed its utility gate and must not be renamed as a successful method.

## 3. Image-level rerun requirements (not bundled or clean-room certified)

Obtain datasets and checkpoints from their authorized sources under their own
terms, preserving source revisions and checksums in the protocol configuration.
Dataset access credentials/passwords are intentionally absent from this release.

- University-1652: https://github.com/layumi/University1652-Baseline
- SUES-200: https://github.com/Reza-Zhu/SUES-200-Benchmark
- DenseUAV: https://github.com/Dmmm1997/DenseUAV
- ViLD and instructions: https://github.com/Tristan-Amadei/caevl and https://zenodo.org/records/19223815
- CAMP source/checkpoint: https://github.com/Mabel0403/CAMP ; source b04a9c856711770ed7a72ebf851838329c5e5b8e
- DenseUAV third-party ViT-S: https://huggingface.co/Bancie/UAV-Self-Positioning-23M-ZCN ; revision ec6fa074268d16cfe655d87d7d086193131ab89a

The original environment was Python 3.12.14, PyTorch 2.8.0+cu128 and torchvision
0.23; CAMP used isolated timm 0.5.4. Exact environment records remain in the
local research project. This file is not a universally tested installation lock.

The historical local project layout is `configs/`, `sources/`, `checkpoints/`,
`data/`, `results/`, `experiments/`. ViLD's authorized extraction belongs at
`data/vild_full/ViLD_dataset/`; its frozen protocol arrays are under
`data/vild_protocol_001/`. Descriptor files order queries first, gallery second.
The original ViLD sequence is `freeze_vild_visual.py` → `infer_vild.py` →
`evaluate_vild.py` → `review_vild.py`. Freeze source, split and hashes in Git
before inference. Exact local commands and input checks are in these scripts.
Original runs must not be overwritten or unlocked by deleting STARTED markers.

Raw image lists, coordinate mappings, descriptor arrays, upstream source trees,
checkpoints and some local manifests are deliberately not redistributed. A new
researcher must prepare these dependencies and a new protocol/run directory;
the release does not currently certify a one-command image-to-table rebuild.
The clean test below covers only Section 1. CAEVL domain-adapted checkpoint
comparison remains pending; downloading ViLD alone does not supply those weights.
