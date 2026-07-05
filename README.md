# Adult Income — Fair & Explainable ML

End-to-end machine learning project predicting whether a person earns **>$50K/year** using the UCI Adult Census Income dataset. Every decision is explained, every finding is traced, and fairness is audited — not as an afterthought, but as a first-class concern.

## Results at a Glance

| Model | Recall (>50K) | Precision | F1 | Accuracy |
|-------|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.84 | 0.58 | 0.69 | 0.81 |
| Random Forest | 0.59 | 0.68 | 0.64 | 0.83 |
| **XGBoost** ✅ | **0.84** | **0.63** | **0.72** | **0.84** |

**Fairness gap:** Male recall 86.6% vs Female recall 69.5% (17pp disparity — documented and flagged).

## Project Structure

```
├── adult-income-fair-explainable-ml.ipynb   # Full notebook (Kaggle-linked)
└── README.md
```

## Approach

The notebook follows four parts, each building on the last:

### Part 1 — Data Understanding & Cleaning
- Discovered hidden nulls encoded as `"?"` (not standard NaN)
- Performed **bias check before dropping** — found null rows are disproportionately female and low-income
- Decision: fill with `"Not-reported"` instead of dropping (prevents bias toward employed/male population)
- Dropped duplicates (24), removed `fnlwgt` and `education` (redundant with `education.num`)

### Part 2 — Exploratory Data Analysis
- **Univariate:** feature distributions, zero-inflation in capital columns
- **Bivariate:** every feature vs income using proportion-based charts (not counts)
- **Feature relationships:** correlation matrix + crosstab revealing `relationship ≈ marital.status` overlap
- **Fairness baseline:** sex (2.7x gap) and race disparities documented before any modeling

### Part 3 — Feature Engineering
- Created binary signals: `has_capital_gain`, `has_capital_loss`, `works_overtime`, `is_us`
- Log-transformed skewed capital columns (`np.log1p`)
- Dropped redundant columns (raw capitals, `marital.status`, `native.country`)
- One-hot encoded categoricals with `drop_first=True`
- Train/test split (80/20, stratified) → StandardScaler fit on train only

### Part 4 — Modeling & Fairness
- Baseline → Logistic Regression → Random Forest → XGBoost
- Evaluated by **Recall and F1** (not accuracy — a 75% baseline is free)
- XGBoost selected: best recall (0.84) + best F1 (0.72)
- **Fairness audit:** measured recall gap across sex groups, documented disparity and mitigation options

## Key Lessons Demonstrated

| Principle | Where it shows up |
|-----------|-------------------|
| Don't drop blindly — check who you're removing | Part 1.4 (bias check) |
| EDA formula: Question → Graph → Finding → Decision | Every Part 2 cell |
| Every engineered feature traces to an EDA finding | Part 3 intro |
| Accuracy is misleading for imbalanced targets | Part 4 model comparison |
| Fairness is measured, not assumed | Part 4.5–4.6 |
| `fit` on train, `transform` on test (no leakage) | Part 3.4 |

## Tech Stack

- **Python 3.12** | pandas | NumPy | matplotlib | seaborn
- **scikit-learn** (Logistic Regression, Random Forest, preprocessing, metrics)
- **XGBoost**
- **Kaggle Notebooks** environment

## Dataset

[UCI Adult Census Income](https://www.kaggle.com/datasets/ucirepository/adult-income) — 32,561 rows, 15 columns. 1994 US Census data.

## How to Run

1. Open the [Kaggle notebook](https://www.kaggle.com/code/abdullahaltepe/adult-income-fair-explainable-ml) and click **"Copy & Edit"**
2. Or clone this repo and run locally:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   jupyter notebook adult-income-fair-explainable-ml.ipynb
   ```

## Future Work

- Per-group threshold tuning to close the fairness gap
- Test `relationship` feature removal (gender proxy)
- Hyperparameter tuning (GridSearchCV / Optuna)
- SHAP values for model explainability
- Deployment-ready pipeline with `ColumnTransformer`

---

*Built as a learning project with emphasis on process quality — not just final accuracy.*
