# Gallery shrinkage in cross-view visual retrieval

Research artifact snapshot: audited experiment tables and the original analysis
scripts used locally. This is ongoing work, not a released safety guarantee or
a claim of state-of-the-art performance. The original SUES historical-SU utility
criterion failed and remains reported. DINOv2 counterexamples are retained.

## What can be verified immediately

Run `python verify_artifacts.py` (Python 3.10+, standard library only). This
checks every exported file hash, table arithmetic, and recorded tie bounds.
It does not repeat image inference or certify scientific novelty.

`tables/stage14/` contains the numerical tables underlying the sixth reviewed
draft. `main_encoder_comparison.csv` is the compact primary comparison.
Detailed tables include infeasible cases; do not silently drop them or compare
means over different feasible masks. Gallery masks are repeated interventions,
not independent places. Fixed-answer risk is not conformal set coverage.

`results/vild_coverage/`, if present, is a metadata-only follow-up, not an image
inference result. It compares whole 500-m cell removal with uniform entry removal
matched to exactly the same entry count, using published reference coordinates.
At 100 m, matched random deletion usually leaves another correct candidate;
spatial deletion removes all support much more frequently. This diagnoses how
to construct a missing-coverage intervention. It does not establish model risk,
novelty, physical accuracy, or natural-weather robustness. No source coordinates,
archive password, or restricted dataset files are included. ViLD data access:
https://zenodo.org/records/19223815 ; official instructions:
https://github.com/Tristan-Amadei/caevl . Cite Amadei et al., Beyond Paired Data:
Self-Supervised UAV Geo-Localization from Reference Imagery Alone, WACV 2026,
pp. 7409–7419, https://arxiv.org/abs/2512.02737 .

## Analysis code and local inputs

`results/vild_visual/`, when present, contains only reviewed real-image CAMP
transfer experiments. Validation and test references are pooled within each
flight; evaluation queries within 500 m of calibration queries are excluded.
The primary working radius is 100 m, with 50/150 m label sensitivity on the same
answers. This is not the original CAEVL benchmark reproduction. The pinned
author TestDataset class defaults to 25 m (overridable by its caller), whereas
the official project reports 100/150/250/500 m and its README includes 100/250 m.
The class default must not be confused with the paper's reporting convention.
`results/vild_25m/`, when present, rescales the labels to 25 m with identical
predictions and accepted queries from the parent 100 m protocol; it does not
fit new thresholds or imply 25 m risk control. The supplementary rule was
declared during image extraction before retrieval outcomes were computed.
Historical-winner survival and historical-confidence scoring are separate
comparators. Both flights spatially overlap; masks/frames are not independent
places, and neither flight is assigned an unverified weather label. Raw images,
coordinates, feature arrays, per-image manifests and passwords are excluded.

The unadapted CAMP model's full-gallery 100 m R@1 is only 208/3754 (5.54%)
and 345/12052 (2.86%) on the two buffered query sets. The historical-score
utility rule fails. These are weak cross-dataset transfer results, not a claim
that the proposed selection mechanism fails for every useful in-domain model.
Author-reported CAEVL numbers use different training and evaluation settings
and are not a controlled comparison against this run. If present,
`results/vild_diagnostic/` checks cached image order, fixed-sample forward passes,
all full-gallery winners, and a post-result test-only-gallery sensitivity.
It changes no frozen outcomes or thresholds and is not original CAEVL replication.

`experiments/` is a byte-identical source snapshot with local import dependencies.
`configs/A-CAMP-*.json` preserves the frozen CAMP settings and provenance hashes.
The historical scripts use a project root containing `results/`, `configs/`,
`sources/` and `checkpoints/`. Many also require a clean committed protocol and
exclusive STARTED markers. Do not delete markers to overwrite an old run.

Image-level reruns additionally require legally obtained data, feature arrays,
upstream source and checkpoint manifests. These are **not included**. This is
an audit snapshot, not a self-contained image-inference distribution. To reuse
the code, acquire the listed sources and prepare the input paths in each config;
do not remove integrity checks to force a run with unverified inputs.

Local computational environment: Python 3.12.14, PyTorch 2.8.0+cu128,
torchvision 0.23, CAMP timm 0.5.4; NumPy, SciPy and OpenCV. CAMP uses FP32,
384x384 RGB, ImageNet normalization, no flips, normalized 1024-D global output.
The Sample4Geo environment used isolated timm 0.9.16. The two supervised
encoders are both ConvNeXt-B, not independent architectures.

## DenseUAV extension

The metadata-first H90 protocol covers all 777 query locations and 3,033 gallery
locations, with both old/plain satellite filename variants. Spatial folds are
defined before scoring, separated by at least 2.88 km. Per-file year mapping
is not independently certified. This tests gallery acquisition change, not
controlled query weather. Identity and 20/50/100 m nominal-coordinate outcomes
are reported separately on identical accepted answers. H90 is a declared slice,
not the complete DenseUAV benchmark. `results/denseuav/`, if present, contains
only results that passed reconstruction review. A subsequent third-party
DenseUAV-trained ViT-S follow-up appears in `results/denseuav_vits/` if reviewed.
It uses the trained 512-D head, baseline test.py preprocessing, and flip summation;
it is not an authenticated author checkpoint. The follow-up is selected after
CAMP results on the same locations. Original failures are retained. Public model:
https://huggingface.co/Bancie/UAV-Self-Positioning-23M-ZCN
revision ec6fa074268d16cfe655d87d7d086193131ab89a.

`results/denseuav_transitions/` contains posthoc ordered eight-cell error
accounting: old gallery, replacement without deletion, replacement plus deletion.
This is descriptive accounting, not unique causal attribution. Infeasible cases
remain in the CSV; summary means use feasible masks and report their count.

## Sources and data access

- University-1652: https://github.com/layumi/University1652-Baseline
  Zheng, Wei, Yang. ACM Multimedia 2020. Research-only dataset; no redistribution.
- CAMP: https://github.com/Mabel0403/CAMP
  source commit b04a9c856711770ed7a72ebf851838329c5e5b8e;
  DOI 10.1109/TGRS.2024.3448499.
- DenseUAV: https://github.com/Dmmm1997/DenseUAV
  DOI 10.1109/TIP.2023.3346279;
  official linked mirror https://huggingface.co/datasets/Dmmm997/DenseUAV,
  revision 0f54323a4ea4d68ae52eb38a5af32bce26955fd1.
- Conformal aerial VPR comparator: DOI 10.1109/ICUAS69441.2026.11598635.
  Our distance/APS experiments adapt its formulas to single-answer selection;
  they do not reproduce the paper's complete original evaluation.

No images, GPS source files, pretrained weights, private correspondence,
unpublished manuscript, personal account paths or local Git history are shipped.
Upstream software and data retain their own licenses. No blanket license is
assigned here to material whose authorship/licensing has not been confirmed.

## Regenerate the paired primary table

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Run `python -I experiments/regenerate_paired_table.py` with Python 3.10+ and no third-party dependencies. This rebuilds paired means from exported aggregate rows, not raw image inference. `results/history_access/` separates cached-query and archive-recomputation costs. `results/history_selector/` preserves the failed exploratory policy-selection preflight.
