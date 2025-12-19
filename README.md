# Reproduction Instructions

This document provides step-by-step instructions to reproduce the experimental results reported in our paper on multilingual news timeline construction.

# News Timeline Construction Pipeline

A three-stage pipeline for constructing news event timelines:

1. **Stage 1**: Pairwise Relatedness Classification (XGBoost)
2. **Stage 2**: Event Clustering (Leiden Algorithm)
3. **Stage 3**: Temporal Ordering (Event Threading)

## Directory Structure

```
final-submit/
├── README.md                    # This file (project documentation)
├── requirements.txt             # Python dependencies
├── data/                        # Raw data files
│   ├── README.md                # Data preparation instructions
│   ├── ETimeline_All_languages.json
│   ├── ETimeline_timeline.json
│   └── chinese.json
├── code/                        # All Python code
│   ├── main_pipeline.py         # Main training/evaluation script
│   ├── data-preperation.py      # Data preprocessing script
│   ├── stage1_pairwise.py       # Stage 1: Pairwise classification
│   ├── stage2_clustering.py     # Stage 2: Event clustering
│   ├── stage3_threading.py      # Stage 3: Temporal threading
│   ├── evaluate_stage3_detailed.py  # Detailed Stage 3 evaluation
│   ├── analyze_linguistic_markers.py  # Linguistic marker analysis
│   ├── visualizations.py        # Figure generation
│   └── language_utils.py        # Multilingual utilities
└── output/                      # Model outputs & results
    ├── figures/                 # Generated visualizations (PDFs & PNGs)
    ├── models/                  # Trained model files (.pkl)
    ├── prepared_data/           # Train/dev/test splits (CSVs)
    └── results/                 # Evaluation results & metrics (JSONs)
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: News Articles                        │
│                    (title, date, url)                               │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Pairwise Relatedness (XGBoost)                           │
│  ─────────────────────────────────────────                         │
│  Input:  Two article titles                                         │
│  Output: P(related) ∈ [0, 1]                                       │
│  Features: TF-IDF + Similarity metrics                              │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Event Clustering (Leiden Algorithm)                       │
│  ─────────────────────────────────────────────                      │
│  Input:  Pairwise similarity scores                                 │
│  Output: Cluster assignments                                        │
│  Method: Temporal windowing → Similarity graph → Community detection│
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Temporal Ordering (Event Threading)                       │
│  ─────────────────────────────────────────────                      │
│  Input:  Articles within a cluster                                  │
│  Output: Ordered timeline with relationship labels                  │
│  Classes: SAME_DAY, IMMEDIATE_UPDATE, SHORT_TERM_DEV, etc.         │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT: Timelines                           │
│           [{position, title, date, role}, ...]                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Install Dependencies

Install core dependencies and optional packages for multilingual support:

```bash
pip install -r requirements.txt
```

The following packages will be installed:
- `pandas`, `numpy`, `scikit-learn`, `xgboost` (core ML)
- `python-igraph`, `leidenalg`, `networkx` (graph clustering)
- `sentence-transformers` (multilingual embeddings)
- `jieba` (Chinese text segmentation)
- `scipy`, `tqdm` (utilities)

**Note**: If you encounter issues with `leidenalg` or `python-igraph`, consult their respective documentation for platform-specific installation instructions (e.g., pre-compiled wheels for Windows, build tools for Linux).

### 2. Data Preparation

Convert the labeled ETimeline JSON dataset into CSV format for training:

```bash
python code/data-preperation.py \
    --labeled data/ETimeline_All_languages.json \
    --output output/prepared_data
```

This will generate the following files in `output/prepared_data/`:
- `pairwise_train.csv` - Training pairs for Stage 1 (pairwise relatedness)
- `pairwise_test.csv` - Test pairs for Stage 1
- `article_index_train.csv` - Article metadata for Stage 2 (clustering)
- `article_index_test.csv` - Test articles for Stage 2
- `threading_train.csv` - Training pairs for Stage 3 (temporal threading)
- `threading_test.csv` - Test pairs for Stage 3

**Expected output**: ~600 topics, ~14K articles, ~5.3M pairwise pairs, ~55K threading pairs.

---

## Reproducing Reported Results

### Baseline: English-Only Pipeline

Train the English-only baseline model (as reported in Table 2 of the paper):

```bash
python code/main_pipeline.py \
    --mode train \
    --data_dir output/prepared_data \
    --model_dir output/models \
    --results_dir output/results \
    --language_mode english
```

**Expected metrics**:
- **Stage 1** (Pairwise): Accuracy 91.30%, F1 91.28%, ROC-AUC 97.47%
- **Stage 2** (Clustering): ARI 84.81%, BCubed F1 80.35%
- **Stage 3** (Threading): Accuracy 37.61%, Macro F1 23.66%

**Output files**:
- `output/models/stage1_xgboost.pkl` - Trained pairwise classifier
- `output/models/stage3_threading.pkl` - Trained threading classifier
- `output/results/training_metrics.json` - All evaluation metrics
- `output/results/stage2_test_clusters.csv` - Predicted clusters

**Training time**: ~15-20 minutes on a modern CPU (8 cores, 3.0 GHz).

---

### Extension 2: Multilingual Pipeline (English + Chinese)

Train the multilingual model with Chinese language support (as reported in Table 3):

```bash
python code/main_pipeline.py \
    --mode train \
    --data_dir output/prepared_data \
    --model_dir output/models \
    --results_dir output/results \
    --language_mode multilingual
```

**Expected metrics**:

| Stage | Metric | English Baseline | Multilingual | Delta |
|-------|--------|------------------|--------------|-------|
| Stage 1 | Accuracy | 91.30% | 92.05% | +0.75% |
| Stage 1 | F1 Score | 91.28% | 92.01% | +0.73% |
| Stage 2 | ARI | 84.81% | 87.55% | +2.74% |
| Stage 2 | BCubed F1 | 80.35% | 83.46% | +3.11% |
| Stage 3 | Accuracy | 37.61% | 28.08% | -9.53% |
| Stage 3 | Macro F1 | 23.66% | 15.82% | -7.84% |

**Per-language Stage 3 performance**:
- English-only pairs: 25.16% accuracy, 13.57% macro F1
- Chinese-only pairs: 31.60% accuracy, 17.57% macro F1 (outperforms English by +6.44%)
- Combined: 28.08% accuracy, 15.82% macro F1

**Output files**:
- `output/models/stage1_xgboost_multilingual.pkl`
- `output/models/stage3_threading_multilingual.pkl`
- `output/results/training_metrics.json` - Overall metrics
- `output/results/stage1_detailed_evaluation.json` - Per-language Stage 1 breakdown
- `output/results/stage2_detailed_evaluation.json` - Per-language Stage 2 breakdown
- `output/results/stage3_detailed_evaluation.json` - Per-language Stage 3 breakdown with per-class metrics
- `output/results/stage2_test_clusters_multilingual.csv`

**Training time**: ~25-30 minutes (larger training set).

---

## Generating Visualizations

After training, generate all figures reported in the paper:

```bash
cd code
python visualizations.py --all
```

This will create the following figures in `output/figures/`:

- `figure_4_roc_curve.pdf` - ROC curve for Stage 1 (AUC 97.44%)
- `figure_5_stage1_confusion_matrix.pdf` - Stage 1 confusion matrix
- `figure_6_stage3_per_class.pdf` - Per-class Stage 3 performance (5 temporal classes)
- `figure_7_stage3_confusion_matrix.pdf` - Stage 3 normalized confusion matrix
- `figure_8_per_language_comparison.pdf` - Stage 2 per-language clustering metrics
- `figure_9_chinese_features.pdf` - Chinese linguistic marker activation rates
- `figure_10_en_vs_multilingual.pdf` - Three-panel comparison (overall, per-language, per-class)
- `figure_11_error_distribution.pdf` - Stage 3 error type breakdown

**To generate individual figures**, use:
```bash
cd code
python visualizations.py --figure <number>  # e.g., --figure 10
```

---

## Evaluation on Test Set

To evaluate trained models without retraining:

```bash
python code/main_pipeline.py \
    --mode evaluate \
    --data_dir output/prepared_data \
    --model_dir output/models \
    --results_dir output/results \
    --language_mode multilingual
```

This will load pre-trained models and compute metrics on the held-out test set.

---

## Advanced: Detailed Per-Language Stage 3 Analysis

For detailed per-class metrics by language (as reported in Section 3.3):

```bash
cd code
python evaluate_stage3_detailed.py
```

**Output**: `output/results/stage3_detailed_evaluation.json` containing precision, recall, F1, and support for each of the 5 temporal classes (`SAME_DAY`, `IMMEDIATE_UPDATE`, `SHORT_TERM_DEV`, `LONG_TERM_DEV`, `DISTANT_RELATED`) broken down by language (English, Chinese, Combined).

**Key findings** (as reported in paper):
- Chinese achieves 0% F1 for `SHORT_TERM_DEV` and `LONG_TERM_DEV` (complete class collapse)
- Both languages over-predict `DISTANT_RELATED` (81.32% recall vs 29.4% true prevalence)

---

## Hyperparameter Configuration

Default hyperparameters (tuned on validation set) are defined in `code/main_pipeline.py`. To modify:

**Stage 1 (Pairwise Classification)**:
```bash
python code/main_pipeline.py --mode train \
    --max_train 100000 \        # Max training pairs (default: 500K)
    --max_test 50000 \          # Max test pairs (default: 50K)
    --no_embeddings             # Disable sentence embeddings (faster, lower accuracy)
```

**Stage 2 (Event Clustering)**:
```bash
python code/main_pipeline.py --mode train \
    --temporal_window 90 \             # Max days between articles in same cluster (default: 90)
    --similarity_threshold 0.3 \       # Min similarity for edge creation (default: 0.3)
    --leiden_resolution 0.8            # Leiden clustering resolution (default: 0.8)
```

---

## Inference on New Articles

To apply trained models to new, unlabeled articles:

1. Prepare a JSON file with article data:
```json
[
  {"title": "Breaking: New policy announced", "date": "2024-01-15"},
  {"title": "Policy details revealed", "date": "2024-01-16"},
  {"title": "Expert analysis of new policy", "date": "2024-01-17"}
]
```

2. Run inference:
```bash
python code/main_pipeline.py \
    --mode inference \
    --input new_articles.json \
    --output predicted_timelines.json \
    --model_dir output/models \
    --language_mode multilingual
```

**Output**: `predicted_timelines.json` containing clustered articles with temporal relation labels.

---

## Troubleshooting

**Out of Memory during training**:
- Reduce `--max_train` and `--max_test` to sample fewer pairs
- Use `--no_embeddings` to disable sentence embeddings (reduces memory by ~50%)

**Low Stage 3 accuracy**:
- This is expected behavior (reported in paper: 28.08% for multilingual)
- Headlines alone lack explicit temporal markers for fine-grained distinctions
- See Error Analysis section (3.4) for detailed explanation

**Leiden clustering errors**:
- Ensure `leidenalg` and `python-igraph` are installed correctly
- On some systems, building from source may be required: `pip install leidenalg --no-binary leidenalg`

**Chinese text not processing**:
- Verify `jieba` is installed: `pip install jieba`
- Check that input JSON contains `language` field set to `"zh"` for Chinese articles

---

## Reproducing Specific Experiments

### Table 2 (Baseline Performance)
```bash
python code/main_pipeline.py --mode train --language_mode english
```

### Table 3 (Multilingual Comparison)
```bash
# Train English baseline
python code/main_pipeline.py --mode train --language_mode english --results_dir output/results_en

# Train multilingual model
python code/main_pipeline.py --mode train --language_mode multilingual --results_dir output/results_multi

# Compare metrics in output/results_en/training_metrics.json and output/results_multi/training_metrics.json
```

### Figure 10 (Three-Panel Comparison)
```bash
# Requires multilingual model trained first
cd code
python visualizations.py --figure 10
```

### Section 3.3 (Chinese Class Collapse Analysis)
```bash
# Train multilingual model
python code/main_pipeline.py --mode train --language_mode multilingual

# Generate detailed per-language breakdown
cd code
python evaluate_stage3_detailed.py

# View per-class metrics
cat output/results/stage3_detailed_evaluation.json | jq '.zh.class_report'
```


## Validation

To verify correct installation and data preparation:

```bash
# Check data files
ls -lh output/prepared_data/*.csv

# Expected output:
# pairwise_train.csv (~3.3M rows)
# pairwise_test.csv (~200K rows)
# threading_train.csv (~47K rows)
# threading_test.csv (~7.6K rows)

# Quick test run (10K samples)
python code/main_pipeline.py \
    --mode train \
    --data_dir output/prepared_data \
    --max_train 10000 \
    --max_test 1000 \
    --language_mode english
```

If this completes successfully (~2 minutes), the full pipeline should run without issues.

---

## Contact

For questions or issues reproducing results, please open an issue in the repository or contact the authors.
