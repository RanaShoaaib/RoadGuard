# RoadGuard

Binary road-anomaly segmentation on AnoVox (CARLA, static mono) with
leave-one-family-out (LOFO) evaluation. Three models per fold — an RGB+depth **teacher**, an RGB
**baseline** and an RGB **student** distilled from the teacher — each available as U-Net or SegFormer.

## Layout

| Path | Purpose |
|---|---|
| `configs/unet.yaml`, `configs/segformer.yaml` | Everything that varies between runs (architecture, family, seed, roles, recipe) |
| `src/roadguard/data/` | Manifest, dataset, transforms, preparation |
| `src/roadguard/models/` | U-Net and SegFormer factory |
| `src/roadguard/training/` | `sampling` (scenario-balanced), `losses` (focal+Dice, distillation), `engine` (epoch, EMA, schedule), `checkpoints`, `config`, `experiment` (the one runner) |
| `src/roadguard/evaluation/` | `metrics` (logit-space pixel/frame/scenario metrics), `reports` (multi-seed/fold tables) |
| `notebooks/` | `01` data preparation, `02` run one fold |

## Run

```python
from roadguard.training.experiment import run_experiment
run_experiment("configs/unet.yaml", fold_csv, data_root, output_root)
```

Tests: `pytest tests`.
