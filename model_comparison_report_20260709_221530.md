# Prescription Exposure — Model Comparison Report

Generated: 2026-07-09 22:15

**Best model (CV-MAE):** Random Forest — 4.53 days

## Performance Metrics

| model             |   mae_days |   rmse_days |   cv_mae_days |   within_7d_pct | top_feature     |   top_feature_imp |
|:------------------|-----------:|------------:|--------------:|----------------:|:----------------|------------------:|
| XGBoost           |       4.44 |        7.88 |          5.39 |              90 | days_supply     |             0.873 |
| Random Forest     |       4.81 |        9.14 |          4.53 |              90 | days_supply     |             0.853 |
| Linear Regression |       6.47 |       11.14 |          6.1  |              75 | adherence_ratio |            71.987 |
| Ridge Regression  |       6.16 |       11.99 |          6.46 |              80 | adherence_ratio |            44.481 |

---

## MLflow

All runs logged to MLflow. To view experiment history and roll back:

```bash
mlflow ui
# Open http://localhost:5000
```

---

## LLM Analysis (Groq)

### 1. RANKING

* **Random Forest**: Ranked as the best model due to its superior performance in cross-validated MAE, which is crucial for predicting prescription exposure end dates accurately in a clinical setting.
* **XGBoost**: Ranked second, offering a good balance between accuracy and the potential for explainability, though slightly outperformed by Random Forest in terms of cross-validated MAE.
* **Ridge Regression**: Ranked third, with its performance being less accurate than the top two models but still offering some degree of interpretability and simplicity.
* **Linear Regression**: Ranked the worst due to its significant underperformance in both MAE and RMSE compared to other models, indicating lower accuracy in predicting prescription end dates.

### 2. PRO/CON TABLE

| Model             | Pros                                                                                       | Cons                                                                                       |
|-------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **XGBoost**        | High accuracy, handles missing values, relatively interpretable                            | Can be complex, requires careful tuning, may not be fully transparent for regulators      |
| **Random Forest**  | High accuracy, robust to missing values, interpretable, handles complex interactions        | Can be computationally intensive, may require significant resources for large datasets    |
| **Linear Regression** | Simple, highly interpretable, easy to implement and understand                            | Low accuracy, assumes linear relationships which may not always hold in clinical data   |
| **Ridge Regression** | Improves upon linear regression by reducing overfitting, still relatively simple         | Less accurate than ensemble methods, may not handle complex interactions as well as others |

### 3. KEY VARIABLES

The models are primarily learning about the relationship between **days_supply** and the prescription exposure end dates, which aligns with clinical expectations since the duration for which a medication is supplied would logically influence when a prescription ends. The **adherence_ratio** also emerges as a significant feature, particularly for linear and ridge regression models, indicating that how consistently a patient takes their medication affects the prescription end date.

### 4. ROLLBACK SCENARIO

If the Random Forest model drifted in production, a significant increase in **cv_mae_days** (beyond a predetermined threshold, e.g., 10% increase) would trigger a rollback decision. In such a case, rolling back to the **XGBoost** model would be a reasonable choice due to its balance between accuracy and interpretability, assuming that the performance metrics of XGBoost remain stable and superior to the other models.

### 5. RECOMMENDATION

I would recommend deploying the **Random Forest** model in a production clinical pipeline due to its high accuracy and robustness to missing values, which are crucial for real-world evidence studies. To ensure governance and regulatory compliance, the following guardrails would be implemented:
- **Model Monitoring**: Continuous monitoring of the model's performance metrics (MAE, RMSE) in production to quickly identify any drift.
- **Explainability Module**: Development of an explainability module to provide insights into the model's decisions, enhancing transparency for regulatory audits.
- **Regular Audits**: Scheduled audits to review the model's performance, data quality, and adherence to regulatory requirements.
- **Update Protocol**: Establishing a protocol for updating the model, including retraining schedules and procedures for incorporating new data or features.

---

*All patient data is synthetic. Analysis generated via Groq LLM.*
*Author: Megha Sharma — github.com/meganalyzer*
