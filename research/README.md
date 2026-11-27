# AI3003 research code and replay records

Run the commands from the repository root with Python 3.11 or later. Install the release and analysis dependencies with `python -m pip install -r research/requirements.txt`.

## Experiment entry points

| Entry point | Purpose |
| --- | --- |
| `research/reference_source/src/prepare_data.py`, `train.py`, `evaluate.py`, `compare.py` | Original Phase I data processing, training, stress evaluation and paired bootstrap. `research/reference_source/scripts/run_experiments.py` runs its four arms. |
| `research/reference_source/scripts/merge_and_check.py` | Original AC42 equal float-weight average and checkpoint/in-domain verification. |
| `scripts/fetch_online_images.py`, `prepare_online_run.py` | Online archive intake, exact SHA-256 screening, dHash candidate quarantine, and dataset-role freeze. |
| `scripts/prepare_expanded_field_trial.py`, `stage_matched_capture_trial.py`, `stage_health_rescue_data.py` | Expanded manifests, capture-group recovery, duplicate exclusion and health/early review data screening. |
| `scripts/train_online_candidate.py train` | Fine-tune AC42's last residual block and classifier. Select the checkpoint on the 37-image Malik validation split using macro-F1. Use the `test` phase only after locking the checkpoint. |
| `scripts/run_subtype_trial.py`, `run_early_review_trial.py`, `run_rescue_stability_trial.py`, `run_directional_subtype_trial.py` | Train and select the successive frozen-feature heads. The last script checks per-group recall, threshold perturbations and the three seeds before separate evaluation. |
| `scripts/run_backbone_trial.py`, `run_hierarchical_trial.py` | Archived ConvNeXt-Tiny/ResNet18 and single-head/hierarchical comparisons. |
| `python -m research.linear_probe fit` and `evaluate` | Standalone three-class DINOv2 linear probe with explicit manifests. |
| `python -m research.paired_statistics` | Recompute paired image-bootstrap intervals and exact McNemar p-values from the archived rows. This path requires no images or model weights. |

The original research pipeline loads its datasets and outputs from `datasets/` and `outputs/` at the checkout root. The historical code uses archived SHA-256 receipts and takes the original course project input directory from `ROBUSTLEAFLAB_REFERENCE` (default `reference/`). `ROBUSTLEAFLAB_PRIOR_WORKSPACE` (default `prior_workspace/`) supplies prior acquisition indices used during screening. The relevant original Python sources are bundled under `research/reference_source/`. The original image archives, intermediate feature caches, and legacy workspaces are external inputs. `--help` documents the available phases and options. Use a separate checkout or an empty outputs directory before executing a fresh historical run.

The immutable protocol and decision JSON files supplied with the course submission describe the completed experiment. Their original checkpoint hashes are included for traceability. Older receipts contain references to original staged manifests. Reconstruct those inputs and paths before attempting an exact rerun. The portable linear-probe and statistics entries do not require the legacy workspace layout.

## Paired replay statistics

Use the archived `research/records/` inputs from the course submission or equivalent locally retained predictions.

```powershell
python -m research.paired_statistics research/records/directional_subtype_trial --audit new_field_audit plantdoc_audit plantseg_subtype_stress taiwan_healthy_late --historical plantcity_replay plantwild_replay imtiaz_replay rachin_replay --output paired_statistics_rerun.json
```

The comparison aligns the v0.1.0 `incumbent` and v0.2.0 `directional_majority` image records by SHA-256 and verifies the reference labels. The image-level percentile bootstrap uses 5,000 replicates and seed 42. The paired statistics JSON supplied with the course submission keeps the audit intervals separate from the first-phase C42/D42 comparison. These intervals are conditional on the archived images and omit capture-group and training-seed variation.

## Three-class frozen DINOv2 probe

The probe uses ViT-S/14 class and mean patch tokens from the released 224 RGB center crop. The fit standardizer and balanced logistic coefficients use 920 training images (354 healthy, 247 early and 319 late). The regularization setting `C=1` was fixed. The 1,378 development images were scored separately. The `evaluate` phase requires a locked decision and checks that no training or development SHA-256 appears in the replay or audit rows. Each manifest needs `path`, `sha256`, and `label` columns.

Use the documented manifest names and hashes in the `probe_config.example.json` supplied with the course submission as a starting point. Supply local image/manifest paths. If re-rooting changes a manifest's bytes, record its new hash only after verifying that the image identities, reference labels, and partition roles are preserved. Use the same configuration for both phases.

```powershell
python -m research.linear_probe fit --config probe_config.json --root . --output linear_probe_run --cache .model-cache --device cuda
python -m research.linear_probe evaluate --config probe_config.json --root . --output linear_probe_run --cache .model-cache --device cuda
```

| Cohort | Linear probe | v0.2.0 |
| --- | ---: | ---: |
| Four historical three-class replays | 752/1,251 (60.11%) | 912/1,251 (72.90%) |
| New field | 307/427 (71.90%) | 191/427 (44.73%) |
| PlantDoc | 181/262 (69.08%) | 196/262 (74.81%) |
| PlantSeg, early/late subset | 88/128 (68.75%) | 91/128 (71.09%) |
| Taiwan, healthy/late subset | 147/204 (72.06%) | 155/204 (75.98%) |

The New field advantage for the probe is specific to the archived source. It highlights a limitation of the released stack under source shift. No audit was used for selection or tuning. Per-image records and the fitted probe are supplied with the course submission.

## Dataset references

- [PlantVillage](https://doi.org/10.3389/fpls.2016.01419), [TOM2024](https://doi.org/10.1016/j.dib.2025.111357), and [Tomato-Village](https://github.com/mamta-joshi-gehlot/Tomato-Village) contributed to historical training.
- [Malik v2](https://data.mendeley.com/datasets/3mbnb82mxd/2) contributed to development, including initial checkpoint selection.
- [PlantCity v4](https://data.mendeley.com/datasets/w8kh2xkspx/4), [PlantWild](https://huggingface.co/datasets/uqtwei2/PlantWild), [Imtiaz](https://data.mendeley.com/datasets/bpfd9cns5g/2), and [Rachin](https://data.mendeley.com/datasets/jttrv2w27r/1) are historical replay cohorts.
- [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) and the archived New field set are post-lock external audits. [PlantSeg](https://zenodo.org/records/17719108) and the Taiwan cohort cover two true classes in this evaluation. See [the release registry](../results/replays.json) for cohort sizes and roles.
