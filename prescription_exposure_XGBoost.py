"""
Prescription Exposure End Date Predictor
-----------------------------------------
Trains an XGBoost regression model on high-confidence rule-based records,
then predicts end dates for ambiguous cases where rules break down.

Training strategy: use rule-engine records with HIGH confidence as labeled ground truth.
The model learns drug class patterns, adherence signals, and seasonal behavior.
Then predicts end dates for low-confidence records the rules engine can't resolve cleanly.

Author: Megha Sharma
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import random

random.seed(42)
np.random.seed(42)

# ─── DRUG CLASS DEFINITIONS ───────────────────────────────────────────────────

DRUG_CONFIG = {
    "Metformin":      {"class": "Biguanide",       "avg_days": 30, "chronic": True,  "std": 5},
    "Lisinopril":     {"class": "ACE Inhibitor",   "avg_days": 30, "chronic": True,  "std": 5},
    "Atorvastatin":   {"class": "Statin",          "avg_days": 30, "chronic": True,  "std": 8},
    "Sertraline":     {"class": "SSRI",            "avg_days": 37, "chronic": True,  "std": 10},
    "Amoxicillin":    {"class": "Antibiotic",      "avg_days": 10, "chronic": False, "std": 2},
    "Warfarin":       {"class": "Anticoagulant",   "avg_days": 30, "chronic": True,  "std": 5},
    "Aspirin":        {"class": "Antiplatelet",    "avg_days": 90, "chronic": True,  "std": 15},
    "Metoprolol":     {"class": "Beta Blocker",    "avg_days": 30, "chronic": True,  "std": 5},
    "Omeprazole":     {"class": "PPI",             "avg_days": 28, "chronic": False, "std": 7},
    "Prednisone":     {"class": "Corticosteroid",  "avg_days": 7,  "chronic": False, "std": 3},
}

INTERACTION_PAIRS = {
    ("Aspirin", "Warfarin"): 0.8,      # 80% chance Aspirin ends when Warfarin starts
    ("Metformin", "Insulin"): 0.7,
}

# ─── SYNTHETIC DATA GENERATOR ─────────────────────────────────────────────────

def generate_synthetic_patients(n=50):
    records = []
    patient_ids = [f"P{str(i).zfill(3)}" for i in range(1, n + 1)]

    for pid in patient_ids:
        # Pick 1-3 drugs per patient
        n_drugs = random.randint(1, 3)
        drugs = random.sample(list(DRUG_CONFIG.keys()), n_drugs)
        fill_base = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 300))

        for drug in drugs:
            config = DRUG_CONFIG[drug]
            n_fills = random.randint(1, 4) if config["chronic"] else 1

            # Adherence ratio — how reliably does this patient refill on time
            adherence = round(random.uniform(0.5, 1.0), 2)

            for fill_num in range(n_fills):
                days_supply = config["avg_days"] + random.randint(-3, 3)
                fill_date = fill_base + timedelta(days=fill_num * int(days_supply / adherence))
                fill_month = fill_date.month

                # True end date — what we're trying to predict
                # For chronic drugs, adherence stretches or compresses the window
                if config["chronic"]:
                    true_duration = int(days_supply / adherence)
                else:
                    true_duration = days_supply + random.randint(0, 3)

                # SSRIs run longer in spring (months 3-5)
                if config["class"] == "SSRI" and fill_month in [3, 4, 5]:
                    true_duration += random.randint(5, 10)

                true_end = fill_date + timedelta(days=true_duration)

                # Check for interaction — does a competing drug shorten this?
                interaction_flag = 0
                for (drug_a, drug_b), prob in INTERACTION_PAIRS.items():
                    if drug == drug_a and drug_b in drugs:
                        if random.random() < prob:
                            true_end = fill_date + timedelta(days=int(days_supply * 0.6))
                            interaction_flag = 1

                # Confidence — is this a high or low confidence record
                has_explicit_end = random.random() < 0.2
                has_refill = (fill_num < n_fills - 1)
                has_days_supply = random.random() < 0.85
                confidence = "high" if (has_explicit_end or has_refill) else "low"

                records.append({
                    "patient_id":           pid,
                    "drug":                 drug,
                    "drug_class":           config["class"],
                    "fill_date":            fill_date,
                    "fill_month":           fill_month,
                    "days_supply":          days_supply if has_days_supply else np.nan,
                    "is_chronic":           int(config["chronic"]),
                    "adherence_ratio":      adherence,
                    "historical_fills":     fill_num + 1,
                    "interaction_flag":     interaction_flag,
                    "has_explicit_end":     int(has_explicit_end),
                    "has_refill_data":      int(has_refill),
                    "confidence":           confidence,
                    "true_end_date":        true_end,
                    "true_duration_days":   true_duration,
                })

    return pd.DataFrame(records)

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def build_features(df):
    le = LabelEncoder()
    df = df.copy()
    df["drug_class_encoded"] = le.fit_transform(df["drug_class"])
    df["days_supply_filled"] = df["days_supply"].fillna(df.groupby("drug_class")["days_supply"].transform("median"))
    return df, le

# ─── MODEL TRAINING ───────────────────────────────────────────────────────────

FEATURE_COLS = [
    "days_supply_filled",
    "drug_class_encoded",
    "fill_month",
    "is_chronic",
    "adherence_ratio",
    "historical_fills",
    "interaction_flag",
]

def train_model(df):
    # Train only on high-confidence records where we trust the labels
    train_df = df[df["confidence"] == "high"].copy()

    print(f"\nTraining on {len(train_df)} high-confidence records out of {len(df)} total")

    X = train_df[FEATURE_COLS]
    y = train_df["true_duration_days"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    print(f"Validation MAE: {mae:.1f} days")

    return model

# ─── PREDICT END DATES ────────────────────────────────────────────────────────

def predict_end_dates(df, model):
    df = df.copy()
    X = df[FEATURE_COLS]
    df["predicted_duration_days"] = model.predict(X).astype(int)
    df["predicted_end_date"] = df.apply(
        lambda r: r["fill_date"] + timedelta(days=int(r["predicted_duration_days"])), axis=1
    )
    df["error_days"] = (df["predicted_end_date"] - df["true_end_date"]).dt.days
    return df

# ─── RUN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating 50 synthetic patients...")
    df = generate_synthetic_patients(n=50)

    df, le = build_features(df)

    model = train_model(df)

    results = predict_end_dates(df, model)

    # ── Summary stats
    print("\n" + "="*70)
    print("MODEL PERFORMANCE ACROSS ALL RECORDS")
    print("="*70)
    overall_mae = results["error_days"].abs().mean()
    low_conf = results[results["confidence"] == "low"]
    high_conf = results[results["confidence"] == "high"]
    print(f"Overall MAE          : {overall_mae:.1f} days")
    print(f"High-confidence MAE  : {high_conf['error_days'].abs().mean():.1f} days")
    print(f"Low-confidence MAE   : {low_conf['error_days'].abs().mean():.1f} days")
    print(f"Records >7 day error : {len(results[results['error_days'].abs() > 7])} ({len(results[results['error_days'].abs() > 7])/len(results)*100:.1f}%)")

    # ── Feature importance
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (what the model learned to rely on)")
    print("="*70)
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 40)
        print(f"  {feat:<30} {bar} {score:.3f}")

    # ── Sample predictions
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (10 records)")
    print("="*70)
    sample = results.sample(10, random_state=42)[["patient_id", "drug", "drug_class",
                                                    "fill_date", "true_end_date",
                                                    "predicted_end_date", "error_days", "confidence"]]
    for _, row in sample.iterrows():
        flag = "REVIEW" if abs(row["error_days"]) > 14 else "OK"
        print(f"{row['patient_id']} | {row['drug']:<14} | True: {row['true_end_date'].date()} | "
              f"Pred: {row['predicted_end_date'].date()} | Error: {row['error_days']:+d}d | {row['confidence'].upper()} | {flag}")

    # ── Save
    results.to_csv("prescription_exposure_XGBoost_results.csv", index=False)
    print(f"\nFull results saved to prescription_exposure_XGBoost_results.csv")
    print(f"Total records: {len(results)}")
