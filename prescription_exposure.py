"""
Prescription Exposure End Date Estimator
-----------------------------------------
For community-dwelling patients (non-facility) picking up prescriptions.

Priority logic for end date estimation:
1. Explicit prescription end date (if recorded)
2. Next refill date (implies prior fill ended)
3. Days supply calculation (fill date + days supply)
4. Assumption-based inference (drug interaction signals, etc.)

Author: Megha Sharma
"""

import pandas as pd
import numpy as np
from datetime import timedelta

# ─── SYNTHETIC DATA ───────────────────────────────────────────────────────────
# Simulating a community prescription fill dataset
# Each row = one prescription fill event

raw_data = [
    # Patient 1: Clean case — explicit end date recorded
    {"patient_id": "P001", "drug": "Metformin",     "fill_date": "2024-01-01", "days_supply": 30, "refill_date": "2024-01-28", "explicit_end_date": "2024-01-30", "drug_class": "Biguanide"},
    {"patient_id": "P001", "drug": "Metformin",     "fill_date": "2024-01-28", "days_supply": 30, "refill_date": "2024-02-25", "explicit_end_date": None,          "drug_class": "Biguanide"},

    # Patient 2: No explicit end date — use refill date
    {"patient_id": "P002", "drug": "Lisinopril",    "fill_date": "2024-01-05", "days_supply": 30, "refill_date": "2024-02-01", "explicit_end_date": None,          "drug_class": "ACE Inhibitor"},
    {"patient_id": "P002", "drug": "Lisinopril",    "fill_date": "2024-02-01", "days_supply": 30, "refill_date": None,         "explicit_end_date": None,          "drug_class": "ACE Inhibitor"},

    # Patient 3: No explicit end, no refill — use days supply only
    {"patient_id": "P003", "drug": "Amoxicillin",   "fill_date": "2024-03-10", "days_supply": 10, "refill_date": None,         "explicit_end_date": None,          "drug_class": "Antibiotic"},

    # Patient 4: Drug interaction signal — starts Warfarin while on Aspirin
    # Aspirin end date unknown — Warfarin start may signal discontinuation
    {"patient_id": "P004", "drug": "Aspirin",        "fill_date": "2024-01-01", "days_supply": 90, "refill_date": None,         "explicit_end_date": None,          "drug_class": "Antiplatelet"},
    {"patient_id": "P004", "drug": "Warfarin",       "fill_date": "2024-02-15", "days_supply": 30, "refill_date": None,         "explicit_end_date": None,          "drug_class": "Anticoagulant"},

    # Patient 5: Gap in refills — possible discontinuation mid-therapy
    {"patient_id": "P005", "drug": "Atorvastatin",  "fill_date": "2024-01-01", "days_supply": 30, "refill_date": "2024-02-01", "explicit_end_date": None,          "drug_class": "Statin"},
    {"patient_id": "P005", "drug": "Atorvastatin",  "fill_date": "2024-02-01", "days_supply": 30, "refill_date": "2024-05-01", "explicit_end_date": None,          "drug_class": "Statin"},  # 90-day gap — likely discontinued then restarted

    # Patient 6: No data at all for end — assumption required
    {"patient_id": "P006", "drug": "Sertraline",    "fill_date": "2024-04-01", "days_supply": None, "refill_date": None,        "explicit_end_date": None,          "drug_class": "SSRI"},
]

# Known drug interaction pairs — starting drug B may signal end of drug A
INTERACTION_PAIRS = {
    ("Aspirin", "Warfarin"): "Antiplatelet typically discontinued when Anticoagulant initiated",
    ("Metformin", "Insulin"): "Oral hypoglycemic often replaced by Insulin",
}

GRACE_PERIOD_DAYS = 7       # Allow 7-day gap before declaring end of exposure
DEFAULT_DAYS_SUPPLY = 30    # Fallback when days_supply is null

# ─── PIPELINE ─────────────────────────────────────────────────────────────────

def estimate_end_dates(data: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df["fill_date"] = pd.to_datetime(df["fill_date"])
    df["refill_date"] = pd.to_datetime(df["refill_date"])
    df["explicit_end_date"] = pd.to_datetime(df["explicit_end_date"])
    df = df.sort_values(["patient_id", "drug", "fill_date"]).reset_index(drop=True)

    results = []

    for _, row in df.iterrows():
        patient_id  = row["patient_id"]
        drug        = row["drug"]
        drug_class  = row["drug_class"]
        fill_date   = row["fill_date"]
        days_supply = row["days_supply"]
        refill_date = row["refill_date"]
        explicit_end = row["explicit_end_date"]

        method      = None
        end_date    = None
        flag        = None

        # ── Priority 1: Explicit end date ──────────────────────────────────
        if pd.notna(explicit_end):
            end_date = explicit_end
            method   = "Explicit end date"

        # ── Priority 2: Refill date ────────────────────────────────────────
        elif pd.notna(refill_date):
            end_date = refill_date - timedelta(days=1)
            method   = "Refill date (next fill - 1 day)"

            # Check for large gap — possible discontinuation
            if days_supply and (refill_date - fill_date).days > (days_supply + GRACE_PERIOD_DAYS):
                flag = f"Gap of {(refill_date - fill_date).days} days exceeds days supply + grace period — possible discontinuation then restart"

        # ── Priority 3: Days supply calculation ────────────────────────────
        elif pd.notna(days_supply):
            end_date = fill_date + timedelta(days=int(days_supply))
            method   = "Days supply calculation"

        # ── Priority 4: Assumption-based ───────────────────────────────────
        else:
            # Check if a conflicting drug was started for this patient
            patient_fills = df[df["patient_id"] == patient_id]
            interaction_found = False

            for (drug_a, drug_b), explanation in INTERACTION_PAIRS.items():
                if drug == drug_a:
                    conflicting = patient_fills[
                        (patient_fills["drug"] == drug_b) &
                        (patient_fills["fill_date"] > fill_date)
                    ]
                    if not conflicting.empty:
                        conflict_date = conflicting["fill_date"].min()
                        end_date = conflict_date - timedelta(days=1)
                        method = f"Assumed end: {drug_b} initiated ({explanation})"
                        flag = "Assumption — drug interaction signal used"
                        interaction_found = True
                        break

            if not interaction_found:
                # Last resort: default days supply
                end_date = fill_date + timedelta(days=DEFAULT_DAYS_SUPPLY)
                method   = f"Default days supply ({DEFAULT_DAYS_SUPPLY} days) — no other signal"
                flag     = "Low confidence — no end date, refill, days supply, or interaction signal available"

        results.append({
            "patient_id":   patient_id,
            "drug":         drug,
            "drug_class":   drug_class,
            "fill_date":    fill_date.date(),
            "estimated_end_date": end_date.date() if pd.notna(end_date) else None,
            "method":       method,
            "flag":         flag or "—"
        })

    return pd.DataFrame(results)


# ─── RUN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = estimate_end_dates(raw_data)

    print("\n" + "="*90)
    print("PRESCRIPTION EXPOSURE END DATE ESTIMATOR — RESULTS")
    print("="*90)

    for _, row in results.iterrows():
        print(f"\nPatient: {row['patient_id']}  |  Drug: {row['drug']} ({row['drug_class']})")
        print(f"  Fill Date:           {row['fill_date']}")
        print(f"  Estimated End Date:  {row['estimated_end_date']}")
        print(f"  Method:              {row['method']}")
        print(f"  Flag:                {row['flag']}")

    print("\n" + "="*90)
    print("FINDINGS SUMMARY")
    print("="*90)
    method_counts = results["method"].value_counts()
    print("\nEnd date estimation methods used:")
    for method, count in method_counts.items():
        print(f"  {count}x  {method}")

    flagged = results[results["flag"] != "—"]
    print(f"\nCases requiring attention or assumptions: {len(flagged)}")
    for _, row in flagged.iterrows():
        print(f"  {row['patient_id']} / {row['drug']}: {row['flag']}")

    # Save to CSV
    results.to_csv("/mnt/user-data/outputs/prescription_exposure_results.csv", index=False)
    print("\nResults saved to prescription_exposure_results.csv")
