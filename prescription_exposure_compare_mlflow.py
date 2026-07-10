"""
Prescription Exposure — Model Comparison Framework with MLflow + Groq LLM
--------------------------------------------------------------------------
Trains multiple models, logs every run to MLflow for versioning and rollback,
compares performance, then sends results to Groq for clinical pro/con analysis.

Author: Megha Sharma

Setup:
    pip install xgboost scikit-learn groq pandas numpy mlflow tabulate

Run:
    export GROQ_API_KEY=your_key_here
    python3 prescription_exposure_compare_mlflow.py

View experiment runs:
    mlflow ui
    Open http://localhost:5000
"""

import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from groq import Groq

random.seed(42)
np.random.seed(42)

# ─── CONFIG — Add or remove models here ───────────────────────────────────────

MODEL_CONFIG = {
    "XGBoost": {
        "model": xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, verbosity=0
        ),
        "params": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1}
    },
    "Random Forest": {
        "model": RandomForestRegressor(n_estimators=100, random_state=42),
        "params": {"n_estimators": 100}
    },
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {}
    },
    "Ridge Regression": {
        "model": Ridge(alpha=1.0),
        "params": {"alpha": 1.0}
    },
}

GROQ_MODEL = "llama-3.3-70b-versatile"
EXPERIMENT   = "prescription_exposure_comparison"
FEATURE_COLS = [
    "days_supply", "drug_class_encoded", "fill_month",
    "is_chronic", "adherence_ratio", "historical_fills", "interaction_flag"
]

# ─── SYNTHETIC DATA ────────────────────────────────────────────────────────────

DRUG_CONFIG = {
    "Metformin":    {"class": "Biguanide",      "avg_days": 30, "chronic": True,  "std": 5},
    "Lisinopril":   {"class": "ACE Inhibitor",  "avg_days": 30, "chronic": True,  "std": 5},
    "Atorvastatin": {"class": "Statin",         "avg_days": 30, "chronic": True,  "std": 8},
    "Sertraline":   {"class": "SSRI",           "avg_days": 37, "chronic": True,  "std": 10},
    "Amoxicillin":  {"class": "Antibiotic",     "avg_days": 10, "chronic": False, "std": 2},
    "Warfarin":     {"class": "Anticoagulant",  "avg_days": 30, "chronic": True,  "std": 5},
    "Aspirin":      {"class": "Antiplatelet",   "avg_days": 90, "chronic": True,  "std": 15},
    "Omeprazole":   {"class": "PPI",            "avg_days": 28, "chronic": False, "std": 7},
    "Prednisone":   {"class": "Corticosteroid", "avg_days": 7,  "chronic": False, "std": 3},
}

def generate_data(n=100):
    records = []
    for i in range(1, n + 1):
        drug   = random.choice(list(DRUG_CONFIG.keys()))
        config = DRUG_CONFIG[drug]
        fill_date   = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 300))
        days_supply = config["avg_days"] + random.randint(-3, 3)
        adherence   = round(random.uniform(0.5, 1.0), 2)
        fill_month  = fill_date.month
        true_dur    = int(days_supply / adherence)
        if config["class"] == "SSRI" and fill_month in [3, 4, 5]:
            true_dur += random.randint(5, 10)
        records.append({
            "drug_class":       config["class"],
            "fill_month":       fill_month,
            "days_supply":      days_supply if random.random() < 0.85 else np.nan,
            "is_chronic":       int(config["chronic"]),
            "adherence_ratio":  adherence,
            "historical_fills": random.randint(1, 5),
            "interaction_flag": int(random.random() < 0.1),
            "true_duration":    true_dur,
        })
    return pd.DataFrame(records)

def prepare_features(df):
    le = LabelEncoder()
    df = df.copy()
    df["drug_class_encoded"] = le.fit_transform(df["drug_class"])
    df["days_supply"] = df["days_supply"].fillna(
        df.groupby("drug_class_encoded")["days_supply"].transform("median")
    )
    return df

# ─── TRAIN, EVALUATE, LOG TO MLFLOW ───────────────────────────────────────────

def run_and_log(df):
    X = df[FEATURE_COLS]
    y = df["true_duration"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment(EXPERIMENT)
    results = []

    for name, config in MODEL_CONFIG.items():
        model  = config["model"]
        params = config["params"]

        with mlflow.start_run(run_name=name):
            # Log parameters
            mlflow.log_param("model_name", name)
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Train
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Metrics
            mae       = mean_absolute_error(y_test, preds)
            rmse      = np.sqrt(mean_squared_error(y_test, preds))
            within_7  = float(np.mean(np.abs(preds - y_test) <= 7) * 100)

            # Cross-validation MAE
            cv_scores = cross_val_score(model, X, y, cv=5,
                                        scoring="neg_mean_absolute_error")
            cv_mae = float(-cv_scores.mean())

            # Log metrics
            mlflow.log_metric("mae",       mae)
            mlflow.log_metric("rmse",      rmse)
            mlflow.log_metric("within_7d", within_7)
            mlflow.log_metric("cv_mae",    cv_mae)

            # Feature importance
            if hasattr(model, "feature_importances_"):
                imp = dict(zip(FEATURE_COLS, model.feature_importances_))
                for feat, score in imp.items():
                    mlflow.log_metric(f"imp_{feat}", float(score))
                top_feature = max(imp, key=imp.get)
                top_score   = imp[top_feature]
            elif hasattr(model, "coef_"):
                imp = dict(zip(FEATURE_COLS, np.abs(model.coef_)))
                top_feature = max(imp, key=imp.get)
                top_score   = imp[top_feature]
            else:
                top_feature = "N/A"
                top_score   = 0.0

            mlflow.log_param("top_feature", top_feature)

            # Log model artifact
            if "XGBoost" in name:
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            print(f"  {name:<20} MAE: {mae:.2f}d  RMSE: {rmse:.2f}d  "
                  f"CV-MAE: {cv_mae:.2f}d  Within 7d: {within_7:.1f}%  "
                  f"Top: {top_feature}")

            results.append({
                "model":           name,
                "mae_days":        round(mae, 2),
                "rmse_days":       round(rmse, 2),
                "cv_mae_days":     round(cv_mae, 2),
                "within_7d_pct":   round(within_7, 1),
                "top_feature":     top_feature,
                "top_feature_imp": round(top_score, 3),
            })

    return pd.DataFrame(results)

# ─── IDENTIFY BEST MODEL ──────────────────────────────────────────────────────

def get_best_model(results_df):
    best = results_df.loc[results_df["cv_mae_days"].idxmin()]
    print(f"\n  Best model by CV-MAE: {best['model']} ({best['cv_mae_days']:.2f} days)")
    print(f"  To roll back: open MLflow UI → select previous run → register that model version")
    return best

# ─── GROQ LLM ANALYSIS ────────────────────────────────────────────────────────

def get_llm_analysis(results_df, best_model_name, api_key):
    client = Groq(api_key=api_key)

    prompt = f"""
You are a clinical data scientist specialising in real-world evidence (RWE) 
and pharmacoepidemiology.

I trained {len(results_df)} models to predict prescription exposure end dates 
for community-dwelling patients — people who pick up prescriptions at a pharmacy.
This is a regression task feeding into drug safety and comparative effectiveness studies.
Explainability matters as much as accuracy for regulatory contexts.

MLflow tracked every run. Best model by cross-validated MAE: {best_model_name}

Results:
{results_df.to_markdown(index=False)}

Provide:

1. RANKING — rank models best to worst for this clinical use case. One sentence each.

2. PRO/CON TABLE — 2-3 pros and 2-3 cons per model specifically for clinical RWE:
   consider accuracy, explainability, missing value handling, regulatory auditability.

3. KEY VARIABLES — what are the models learning about prescription behaviour? 
   Does the top feature align with clinical expectations?

4. ROLLBACK SCENARIO — if the best model drifted in production and you needed 
   to roll back, what metric would trigger that decision and what would you roll back to?

5. RECOMMENDATION — which model would you deploy in a production clinical pipeline 
   and why? What governance guardrails would you put around it?

Be concise, specific, and clinical. No generic AI text.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1800,
    )
    return response.choices[0].message.content

# ─── SAVE REPORT ──────────────────────────────────────────────────────────────

def save_report(results_df, best, llm_analysis):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path  = f"model_comparison_report_{ts}.md"
    csv_path = f"model_comparison_results_{ts}.csv"

    results_df.to_csv(csv_path, index=False)

    with open(md_path, "w") as f:
        f.write("# Prescription Exposure — Model Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Best model (CV-MAE):** {best['model']} — {best['cv_mae_days']} days\n\n")
        f.write("## Performance Metrics\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## MLflow\n\n")
        f.write("All runs logged to MLflow. To view experiment history and roll back:\n\n")
        f.write("```bash\nmlflow ui\n# Open http://localhost:5000\n```\n\n")
        f.write("---\n\n")
        f.write("## LLM Analysis (Groq)\n\n")
        f.write(llm_analysis)
        f.write("\n\n---\n\n")
        f.write("*All patient data is synthetic. Analysis generated via Groq LLM.*\n")
        f.write("*Author: Megha Sharma — github.com/meganalyzer*\n")

    return md_path, csv_path

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = input("Enter your Groq API key: ").strip()

    print("\nGenerating synthetic dataset (100 patients)...")
    df = generate_data(n=100)
    df = prepare_features(df)

    print(f"\nTraining {len(MODEL_CONFIG)} models and logging to MLflow...\n")
    results_df = run_and_log(df)

    best = get_best_model(results_df)

    print("\nSending results to Groq for clinical analysis...")
    llm_analysis = get_llm_analysis(results_df, best["model"], api_key)

    print("\n" + "="*70)
    print("LLM ANALYSIS")
    print("="*70)
    print(llm_analysis)

    md_path, csv_path = save_report(results_df, best, llm_analysis)

    print("\n" + "="*70)
    print(f"Report:  {md_path}")
    print(f"Metrics: {csv_path}")
    print("\nTo view all MLflow runs:")
    print("  mlflow ui")
    print("  Open http://localhost:5000")
