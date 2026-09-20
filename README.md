# Mango Variety Recognition Using Lightweight Transfer Learning


## Team

| Name | ID 
|---|---
| Miraj Mahmud Mahee | 210041101 
| Mueed Ibne Sami | 210041149
| Khandaker Saif Karim | 210041151
| Mst Sumaiya Tasnim | 210041154


## Summary

We fine-tune pretrained CNNs on the MangoClassify-12 dataset (3,900 phone photos, 12 varieties, from 41 to 600 images per class). Our research question: **does class-balanced training (class weights, focal loss, augmentation) improve recognition of rare varieties compared with plain fine-tuning?**

Main findings:

- Transfer learning and fine-tuning give the biggest gains: small CNN from scratch 0.771 macro-F1, frozen MobileNetV2 0.838, fine-tuned MobileNetV2 0.961.
- The best model is plain fine-tuned MobileNetV2 (no augmentation, no class weights): **macro-F1 0.962 ± 0.001** over 3 seeds.
- Our class-balanced model (augmentation + class weights) was slightly worse: 0.937 ± 0.011 over 3 seeds. Our hypothesis that class weighting helps rare varieties was **not supported**. The rarest classes were already recognised well; the hardest class was Fazli (often predicted as Langra).
- Rare-class test sets are tiny (6 to 18 images), so rare-class conclusions are tentative.

## Dataset

MangoClassify-12 (Kaggle): https://www.kaggle.com/datasets/researchersajid/mangoclassify-12-native-mango-dataset-from-bd



Images per class: Amrapali 600, Harivanga 575, Langra 506, Himsagar 502, GopalBhog 406, Khrishapat 380, Bari 4 240, Sundari 226, Banana 212, Fazli 120, RaniBhog 92, GobindoBhog 41.

## Method

**Pipeline**

1. Read all images, resize to 256x256 (EXIF rotation fixed) and cache.
2. Extract 1,280-dimensional features with a pretrained MobileNetV2.
3. **Leakage protection:** the dataset has no fruit IDs, so near-duplicate photos are grouped by agglomerative clustering (cosine distance threshold 0.08) within each class: 1,504 groups, largest 51 images.
4. **Group-aware split** with `StratifiedGroupKFold` (7 folds: 1 test, 1 validation, 5 train). Among 60 random seeds, the split with the most even class proportions was kept (seed 58). This was chosen using class balance only, never model scores. Split: 2,779 train / 550 validation / 571 test.
5. Train models, choose the best epoch by **validation macro-F1**, and evaluate on the test set once.

**Models:** small CNN (from scratch); ImageNet-pretrained MobileNetV2, ResNet18, EfficientNet-B0 with a new 12-class head.

**Training:** Adam, batch size 32, 224x224 input, 20 epochs (CNN) / 12 epochs (pretrained). Fine-tuning: 3 epochs head only (lr 1e-3), then upper blocks unfrozen (lr 1e-4).

**Losses:** cross-entropy; class-weighted cross-entropy (weights proportional to 1/sqrt(class count)); focal loss (gamma = 2).

**Augmentation (training only):** random resized crop, horizontal flip, rotation up to 15 degrees, brightness jitter.

**Metrics:** accuracy, macro-precision, macro-recall, macro-F1 (main metric), per-class recall and F1, confusion matrix.

## Experiments

| ID | Model | Training | Augmentation | Class weights |
|---|---|---|---|---|
| E1 | Small CNN | from scratch | no | no |
| E2 | MobileNetV2 | frozen | no | no |
| E3 | MobileNetV2 | fine-tuned | no | no |
| E4 | MobileNetV2 | fine-tuned | yes | no |
| E5 | MobileNetV2 | fine-tuned | no | yes |
| E6 | MobileNetV2 | fine-tuned | yes | yes |
| E7 | ResNet18 / EfficientNet-B0 | fine-tuned | yes | yes |
| E8 | MobileNetV2 (focal loss) | fine-tuned | yes | no |

E3 and E6 were repeated with seeds 42, 1, and 2.

## Results (test set, seed 42 unless stated)

| ID | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|
| E1 | 0.790 | 0.808 | 0.781 | 0.771 |
| E2 | 0.876 | 0.878 | 0.824 | 0.838 |
| **E3** | **0.970** | 0.972 | 0.952 | **0.961** |
| E4 | 0.946 | 0.966 | 0.904 | 0.920 |
| E5 | 0.963 | 0.950 | 0.949 | 0.946 |
| E6 | 0.955 | 0.972 | 0.917 | 0.930 |
| E7 EfficientNet-B0 | 0.963 | n/a | n/a | 0.952 |
| E7 ResNet18 | 0.921 | n/a | n/a | 0.903 |
| E8 focal loss | 0.942 | 0.951 | 0.887 | 0.912 |

Three seeds (macro-F1): E3 = 0.962 ± 0.001, E6 = 0.937 ± 0.011.

Per-class recall (E3 vs E6, mean over 3 seeds) shows that most of E6's loss comes from **Fazli** (0.667 to 0.444) and **Bari 4** (0.968 to 0.860), while the rarest classes (GobindoBhog, RaniBhog) were already recognised well. Seed-42 ablations suggest augmentation, not class weights, as the likelier cause, but E4 and E5 are single runs, so this is a hypothesis, not a proven cause.



**Efficiency** 

| Model | Parameters (M) | Size (MB) | ms per image |
|---|---|---|---|
| Small CNN | .39 | 1.6 | 1.13 |
| MobileNetV2 | 2.24 | 9.2 | 5.01 |



## Limitations

- Near-duplicate grouping is approximate, so some photos of the same fruit may still be split across train and test. Scores may be optimistic.
- Rare-class test sets are very small (GobindoBhog 6, Fazli 12), so their recall is noisy.
- All seeds share the same test set; seeds measure training randomness, not test-set variation.
- Only E3 and E6 have multiple seeds; other experiments are single runs.
- We do not know why augmentation hurt; explanations are untested hypotheses.
- Single dataset, no outside test photos, and no on-phone deployment.

## How to run

1. Open `mango_pipeline.ipynb` on [Kaggle](https://www.kaggle.com) and attach the dataset linked above.
2. In the session settings, turn on **GPU** and **Internet** (needed to download pretrained weights).
3. Run all cells top to bottom. Results are written to `/kaggle/working/outputs`.

## Repository structure

```
.
├── README.md
├── mango_pipeline.ipynb        full code and outputs
├── outputs/
│   ├── CSV/
│   │   ├── main_results.csv
│   │   ├── extension_results.csv
│   │   ├── per_class_E3_vs_E6.csv
│   │   ├── efficiency.csv
│   │   └── split_counts_per_class.csv
│   └── PNG/
│       ├── confusion_E3_E6.png
│       ├── training_curves.png
│       ├── class_distribution.png
│       └── gradcam.png
└── report PDF
```

