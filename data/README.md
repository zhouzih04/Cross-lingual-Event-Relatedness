# Data Directory

This directory contains the raw ETimeline dataset files:

- `ETimeline_All_languages.json` - Full multilingual timeline dataset (English + Chinese)
- `ETimeline_timeline.json` - Timeline annotations
- `chinese.json` - Chinese language subset

## Generating Train/Dev/Test Splits

To generate the training, validation, and test splits used in this project, run the data preparation script from the project root:

```bash
python code/data-preperation.py \
    --labeled data/ETimeline_All_languages.json \
    --output output/prepared_data
```

Or from the `code/` directory:

```bash
cd code
python data-preperation.py \
    --labeled ../data/ETimeline_All_languages.json \
    --output ../output/prepared_data
```

This will create processed data files in the `output/prepared_data/` directory, including:
- `pairwise_train.csv`, `pairwise_val.csv`, `pairwise_test.csv` - Pairwise comparison data for Stage 1
- `article_index_train.csv`, `article_index_val.csv`, `article_index_test.csv` - Article indices for Stage 2 clustering
- `threading_train.csv`, `threading_val.csv`, `threading_test.csv` - Threading data for Stage 3
- `xgboost_train.csv`, `xgboost_val.csv`, `xgboost_test.csv` - XGBoost-compatible format
- `clustering_constraints.json` - Must-link/cannot-link constraints
- `topic_splits.json` - Topic split assignments

Refer to `data-preperation.py` in the `code/` directory for details on the splitting methodology and feature extraction.
