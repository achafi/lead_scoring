# Lead Scoring Project Report

## 1. Project Objective

The goal is to estimate whether a lead will convert. A lead score can help a sales team rank prospects and decide which leads to contact first. The score supports prioritization; it does not guarantee a conversion.

## 2. Data and EDA

The project uses the X Education Kaggle dataset with **9,240 leads and 37 columns**. The target, `Converted`, indicates whether a lead converted: **3,561 leads (38.5%) converted** and **5,679 (61.5%) did not**. The dataset has no exact duplicate rows; `Prospect ID` and `Lead Number` are unique identifiers.

Blank values and the form placeholder `Select` are treated as missing/not selected. Missingness is substantial: `Lead Quality` is missing for 51.6% of leads, asymmetrique fields for 45.6%, `Tags` for 36.3%, and `Country` for 26.6%. Website engagement was associated with conversion: median time on the website was 832 seconds for converted leads and 179 seconds for non-converted leads. Conversion rates also differed across lead sources and occupations.

The model excludes IDs, constant fields, `Tags`, `Lead Quality`, `Lead Profile`, and `Last Notable Activity`. IDs cannot generalize to new leads; status and quality fields could contain information added during or after follow-up. `Last Activity` and asymmetrique activity/profile scores remain in the model, but their availability at the intended scoring time has not been verified.

## 3. Modeling Approach

The notebook replaces blanks and `Select` with missing values. Numeric fields are median-imputed, scaled, and given missing-value indicators. Categorical fields are filled with a `Missing` category and one-hot encoded; rare categories are grouped by the encoder. The split is stratified into 80% training and 20% test data.

The candidates were a majority-class baseline, logistic regression, random forest, histogram gradient boosting, XGBoost, and LightGBM. The final model was selected by the highest mean PR-AUC in five-fold stratified cross-validation on the training set. **XGBoost** was selected with mean PR-AUC 0.897, narrowly ahead of LightGBM at 0.895.

## 4. Model Evaluation

Test-set results for the selected model:

| Metric | Result | Meaning for this project |
|---|---:|---|
| Accuracy | 0.835 | 83.5% of test predictions were correct at the default threshold. |
| Precision | 0.759 | 75.9% of leads flagged as converted did convert in the test set. |
| Recall | 0.837 | The model identified 83.7% of the converted test leads. |
| F1-score | 0.796 | Balances precision and recall at the default threshold. |
| ROC-AUC | 0.914 | The model separates converted from non-converted leads well across score thresholds. |
| PR-AUC | 0.871 | Measures how well the model ranks converted leads, with focus on the positive class. |

Five-fold cross-validation results for XGBoost:

| Metric | Mean | Standard deviation |
|---|---:|---:|
| ROC-AUC | 0.928 | 0.005 |
| PR-AUC | 0.897 | 0.012 |

The small standard deviations indicate that the AUC results varied little across these folds. XGBoost narrowly led cross-validation PR-AUC; LightGBM had slightly lower fold variation and marginally higher test PR-AUC, recall, and F1. The test results are broadly consistent with cross-validation. These are results on a random split of this dataset, not a guarantee of future performance.

## 5. Lead Scoring

The final model produces `final_score`, an estimated conversion probability for each test lead. For example, a score of **0.72** means the model estimates a 72% conversion probability for that lead. It is a model estimate, not a promise or necessarily a perfectly calibrated probability.

## 6. Lead Segmentation

The test leads were divided using the one-third and two-thirds quantiles of their predicted scores. The resulting cutoffs were approximately **0.126** and **0.723**. This creates relative groups of similar size rather than using arbitrary fixed probability cutoffs.

| Segment | Score range | Leads | Share | Average predicted probability | Actual conversion rate |
|---|---|---:|---:|---:|---:|
| Low | ≤ 0.126 | 625 | 33.8% | 4.5% | 3.4% |
| Medium | > 0.126 to ≤ 0.723 | 607 | 32.8% | 37.4% | 29.0% |
| High | > 0.723 | 616 | 33.3% | 90.5% | 83.6% |

Observed conversion rates rise from Low to Medium to High. On this test set, higher-scored leads therefore had higher observed conversion rates. The test labels were used to check the segments, not to set the score cutoffs.

## 7. SHAP Explainability

SHAP ranked `Last Activity` and `Total Time Spent on Website` as the largest contributors overall, followed by `Lead Origin`, `Asymmetrique Activity Score`, course preference, and occupation. For one high-score lead, `Lead Origin_Lead Add Form`, website time, and activity score pushed the output toward conversion. For one low-score lead, `Last Activity_Converted to Lead` pushed it away from conversion. XGBoost SHAP contributions are in raw-score (log-odds) units; they describe model behavior, not causal effects.

The activity fields need a timing check: if they are recorded after initial lead intake or during sales follow-up, they may not be available when the lead should first be scored. SHAP describes model associations and does not show that a feature causes conversion.

## 8. Business Use

- **High propensity:** prioritize for timely sales follow-up.
- **Medium propensity:** use the normal follow-up process.
- **Low propensity:** give lower immediate priority or use a suitable nurturing process.

The actual threshold between groups should reflect sales capacity and the cost of missing likely customers versus contacting leads that do not convert.

## 9. Limitations

- Results depend on this public dataset and may not transfer to another company or future leads.
- Scores are estimates, not guarantees; probability calibration should be checked before interpreting them as precise chances.
- Quantile segments are relative to this test-score distribution. New leads may have a different score distribution, so the cutoffs may need review.
- Timing is unverified for `Last Activity` and the asymmetrique fields. Confirm these are available at the moment the score will be used.
- Validation on later, unseen leads and on the client’s own CRM data would provide a stronger estimate of operational performance.

## 10. Final Conclusion

This project builds a workflow that estimates conversion probability, ranks and segments leads, and explains model predictions. XGBoost had the highest mean cross-validated PR-AUC and separated conversion outcomes well on the held-out test set, though its advantage over LightGBM was small. The test segments also showed increasing observed conversion rates from Low to High, giving a practical starting point for lead prioritization. Timing checks and validation on future or client-specific data are needed before operational use.
