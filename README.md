# Neural Network-based Recommendation System

A hybrid neural network approach to solve the cold-start problem in recommendation systems using demographic and content-based features.

## Project Overview

This project implements a Hybrid Neural Network recommendation system that addresses the **cold-start problem** for new users by leveraging:
- User demographic information (age, gender, occupation, location)
- Item content features (genres, metadata)
- Embedding layers with Multi-Layer Perceptron (MLP)

### Hypothesis
> The Hybrid Neural Network will achieve **Precision@10 ≥ 0.35** and **Hit Rate ≥ 0.70** for cold-start users, outperforming standard NCF by at least 3%.

## Project Structure

```
nn-recommendation-system/
│
├── README.md
├── requirements.txt
├── config.py
│
├── data/
│   ├── raw/                    # Original MovieLens data
│   │   └── .gitkeep
│   ├── processed/              # Cleaned and processed data
│   │   └── .gitkeep
│   └── download_data.py        # Script to download MovieLens
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Load and preprocess data
│   │   └── feature_engineer.py # Feature engineering
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py         # Random & Popularity baselines
│   │   ├── matrix_factorization.py
│   │   ├── ncf.py              # Neural Collaborative Filtering
│   │   └── hybrid_nn.py        # Our proposed model
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # Precision, Recall, NDCG, etc.
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── experiments/
│   ├── run_experiment.py       # Main experiment runner
│   └── results/                # Saved results and models
│       └── .gitkeep
│
└── tests/
    ├── __init__.py
    └── test_models.py
```

## Project Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/nn-recommendation-system.git
cd nn-recommendation-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Data

```bash
python data/download_data.py
```

### 4. Run Experiments

```bash
python experiments/run_experiment.py
```

### 5. View Results in Notebooks

```bash
jupyter notebook notebooks/
```

## Requirements

```text
# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
torch>=1.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
tqdm>=4.61.0
```

## Dataset

**MovieLens 100K** dataset containing:
- 100,000 ratings (1-5)
- 943 users
- 1,682 movies
- User demographics: age, gender, occupation, zip code
- Movie genres: 19 genres

Download: https://grouplens.org/datasets/movielens/100k/

## Methodology

### Pipeline Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   1. Data    │───▶│  2. Feature  │───▶│  3. Model    │
│  Collection  │    │  Engineering │    │   Training   │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                                               ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 6. Present   │◀───│  5. Analysis │◀───│ 4. Testing   │
│   Results    │    │              │    │  & Metrics   │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Models Compared

| Model | Description |
|-------|-------------|
| Random | Random recommendations (baseline) |
| Popularity | Most popular items (baseline) |
| Matrix Factorization | Traditional collaborative filtering |
| Standard NCF | Neural Collaborative Filtering |
| **Hybrid NN (Ours)** | Demographic + Content + Neural |

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Precision@10 | ≥ 0.35 | Relevant items in top-10 |
| Recall@10 | ≥ 0.25 | Coverage of relevant items |
| NDCG@10 | ≥ 0.40 | Ranking quality |
| Hit Rate | ≥ 0.70 | At least 1 relevant in top-10 |
| RMSE | ≤ 0.90 | Rating prediction error |

## Experiments

### Run All Models

```bash
python experiments/run_experiment.py --all
```

### Run Specific Model

```bash
python experiments/run_experiment.py --model hybrid_nn
```

### Hyperparameter Tuning

```bash
python experiments/run_experiment.py --model hybrid_nn --tune
```

## Expected Results

| Model | Precision@10 | RMSE | Cold-Start |
|-------|--------------|------|------------|
| Random | 0.05 | 1.50 | ✗ |
| Popularity | 0.15 | 1.20 | ✗ |
| Matrix Factorization | 0.30 | 0.95 | ✗ |
| Standard NCF | 0.35 | 0.88 | ✗ |
| **Hybrid NN** | **0.38** | **0.85** | **✓** |

## Key Files Description

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters and settings |
| `src/models/hybrid_nn.py` | Main proposed model |
| `src/evaluation/metrics.py` | All evaluation functions |
| `experiments/run_experiment.py` | Run and compare all models |
| `notebooks/04_evaluation_analysis.ipynb` | Results visualization |

## Configuration

Edit `config.py` to modify:

```python
# config.py
CONFIG = {
    'embedding_dim': 32,
    'hidden_layers': [64, 32],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 100,
    'early_stopping_patience': 10
}
```

## Usage Example

```python
from src.data.data_loader import load_movielens
from src.models.hybrid_nn import HybridNN
from src.evaluation.metrics import evaluate_model

# Load data
train, test, user_features, item_features = load_movielens()

# Train model
model = HybridNN(config)
model.fit(train, user_features, item_features)

# Evaluate
results = evaluate_model(model, test, k=10)
print(f"Precision@10: {results['precision']:.4f}")
```

## Results Visualization

Results are saved in `experiments/results/` and can be visualized in the evaluation notebook.

## References

- He, X., et al. (2017). Neural Collaborative Filtering
- Cheng, H. T., et al. (2016). Wide & Deep Learning
- Sedhain, S., et al. (2015). AutoRec
- Guo, H., et al. (2017). DeepFM
- Volkovs, M., et al. (2017). DropoutNet

## Author

Chinzorigt Ganbat - [@dschinzo](https://github.com/dschinzo)