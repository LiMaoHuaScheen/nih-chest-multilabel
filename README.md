# NIH Chest X-ray Multi-Label Classification

Fine-tuning DenseNet-121 (ImageNet-pretrained) on a 7,491-image subset of NIH ChestX-ray14 for 14-disease multi-label classification. Compares three loss / sampling strategies: BCE baseline, Focal Loss, and WeightedRandomSampler.

## Results (test set, n=1125)

| Arm | Test macro-AUC | Test macro-F1 (thr=0.5) | Best epoch |
|---|---|---|---|
| baseline (BCE) | **0.7427** | 0.0805 | 4 / 8 |
| focal (alpha=0.25, gamma=2) | 0.7387 | **0.1006** | 8 / 8 |
| weighted sampler | 0.7125 | 0.0623 | 2 / 8 |

Macro-AUC is the primary metric (threshold-free); macro-F1 uses a naive 0.5 threshold and is expected to be low on this class-imbalanced multi-label task.

Per-class AUCs (baseline arm) range from 0.86 (Edema) down to 0.64 (Fibrosis). Fibrosis, Effusion, and Pneumothorax are the hardest classes.

## Figures

- `figures/curve_auc.png` - validation macro-AUC per epoch for all three arms
- `figures/bar_arms.png` - test macro-AUC and macro-F1 for all three arms
- `figures/bar_per_class.png` - per-class test AUC for all three arms
- `figures/gradcam.png` - Grad-CAM heatmaps on held-out test images

## Setup

```bash
pip install -r requirements.txt
python code/split_data.py      # random stratified 70/15/15 split
python code/train_chest.py     # train baseline / focal / weighted arms
python code/plot_chest.py      # generate figures from results
python code/gradcam.py         # Grad-CAM heatmaps from baseline_best.pt
```

## Dataset

This repo does not ship the images. `code/extract_images.py` (not included) shows how the 7,491 PNG images were extracted from the `g-ronimo/NIH-Chest-X-ray-dataset_10k` Hugging Face subset. Run `code/split_data.py` after placing images and labels.csv under `chest_data/`.

## Honest limitations

- **No patient-wise split.** The 10k subset has no `patient_id`, so a random stratified 70/15/15 split is used; some patient-level leakage across train/val/test is possible.
- **No negative class.** All 7,491 images contain at least one of the 14 diseases; there is no "No Finding" image, so the model never sees a true negative.
- **Small subset.** 7,491 images vs ~112k in the full NIH ChestX-ray14; per-class AUCs are correspondingly lower than published full-dataset baselines.
- **Possible shortcut learning.** Grad-CAM heatmaps on some test images highlight the radiology annotation corner (e.g., "PORTABLE" text) rather than lung pathology, suggesting the model may exploit scanner / view markers as spurious cues.

## License

MIT
