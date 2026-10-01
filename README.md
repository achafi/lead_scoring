# Lead Scoring Machine Learning Project

## Project Overview

This project predicts the probability that a lead will convert. The model assigns each lead a score for prioritization, groups scored leads into Low, Medium, and High propensity segments, and uses SHAP to explain the model's predictions.

## Dataset

The project uses the X Education Kaggle lead dataset, with **9,240 rows and 37 columns**. `Converted` is the binary target. The dataset has no exact duplicate rows, while `Prospect ID` and `Lead Number` are unique identifiers. Blank cells and `Select` placeholders are treated as missing/not selected. Several fields have substantial missingness.

## Project Workflow

**Data → EDA → Preprocessing → Model Training → Model Evaluation → Cross-Validation → SHAP → Lead Scoring → Segmentation**

## Models

The notebook compared a majority-class baseline, logistic regression, random forest, histogram gradient boosting, XGBoost, and LightGBM. **XGBoost** was selected for the highest mean PR-AUC in five-fold stratified cross-validation on the training split (0.897 versus 0.895 for LightGBM). The difference is small: LightGBM had slightly lower fold variation and marginally higher test PR-AUC, recall, and F1.

## Model Evaluation

| Final test metric | Result |
|---|---:|
| Precision | 0.759 |
| Recall | 0.837 |
| F1 | 0.796 |
| ROC-AUC | 0.914 |
| PR-AUC | 0.871 |

| Five-fold cross-validation metric | Mean | Standard deviation |
|---|---:|---:|
| ROC-AUC | 0.928 | 0.005 |
| PR-AUC | 0.897 | 0.012 |

Precision is the share of leads flagged by the model that converted in the test set. Recall is the share of converted leads the model identified. F1 balances the two. ROC-AUC measures separation across thresholds; PR-AUC focuses on ranking the converted leads.

## Explainability

SHAP was used to explain the selected model. XGBoost SHAP contributions are in raw-score (log-odds) units; positive contributions push the model output toward conversion. The largest overall contributors were `Last Activity`, `Total Time Spent on Website`, and `Lead Origin`, followed by `Asymmetrique Activity Score`, course preference, and occupation. For one high-score lead, `Lead Origin_Lead Add Form`, website time, and activity score pushed the model output toward conversion. SHAP describes model associations, not causes.

## Lead Scoring

The model produces `final_score`, its estimated conversion probability for each scored lead. For example, `0.72` is a model estimate of 72% conversion probability, not a guarantee.

## Lead Segmentation

Test leads were grouped using the one-third and two-thirds quantiles of their scores, with cutoffs of approximately **0.126** and **0.723**. These are relative score-distribution cutoffs, not universal probability thresholds.

| Segment | Leads | Share | Average predicted probability | Actual conversion rate |
|---|---:|---:|---:|---:|
| Low | 625 | 33.8% | 4.5% | 3.4% |
| Medium | 607 | 32.8% | 37.4% | 29.0% |
| High | 616 | 33.3% | 90.5% | 83.6% |

Observed conversion rates increase from Low to High on this held-out test set.

## Business Use

- **High propensity:** higher priority for sales follow-up.
- **Medium propensity:** normal follow-up.
- **Low propensity:** lower immediate priority or nurturing.

The team should choose operational cutoffs based on its capacity and follow-up goals.

## Limitations

- Results depend on this public dataset and may not transfer to another business or future leads.
- Predicted probabilities are model estimates, not guarantees; calibration should be checked before using scores as precise probabilities.
- Quantile-based segments are relative to the test-score distribution and may need review for future data.
- The timing of `Last Activity` and the asymmetrique activity/profile fields is not verified. Confirm they are available at the intended scoring point.
- Future or client-specific CRM data would provide stronger validation for operational use.

## Project Structure

```text
lead_scoring/
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Modeling.ipynb
├── reports/
│   └── lead_scoring_project_report.md
├── models/
│   ├── final_model.joblib
│   └── model_metadata.json
├── src/
│   └── scoring.py
├── data/
│   ├── Lead Scoring.csv
│   ├── Leads Data Dictionary.xlsx
│   └── README.md
└── README.md
```

## Run the Notebooks and Scoring Script

The data files are not included in this repository. Download the X Education Lead Scoring dataset from Kaggle and place Lead Scoring.csv in the data/ folder. Place Leads Data Dictionary.xlsx there too if you want to reference the column descriptions.

From the project root, create an environment and install the notebook dependencies:

```bash
uv venv
uv pip install --python .venv/bin/python jupyterlab pandas numpy matplotlib seaborn scikit-learn==1.9.1 shap joblib xgboost==3.4.1 lightgbm==4.7.0
.venv/bin/jupyter lab
```

To score a CSV with the required model input columns:

```bash
.venv/bin/python src/scoring.py --input new_leads.csv --output scored_leads.csv
```

The output preserves the input columns and adds `final_score` and `lead_segment`. Use `--no-segmentation` to return scores only. The scoring script loads the saved pipeline; it does not train a model.
