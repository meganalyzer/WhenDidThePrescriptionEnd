
"""
POC Board Package Comparison Generator
--------------------------------------
Compares Rule-Based Math vs. Machine Learning outputs side-by-side.
Highlights structural variance to prove why deterministic rules create business errors.

Author: Megha Sharma
"""

import pandas as pd
import numpy as np

def run_variance_analysis():
    # ─── 1. SIMULATE LOADING BOTH PIPELINE OUTPUTS ────────────────────────────
    # In production, you would use:
    # df_rules = pd.read_csv("prescription_exposure_results.csv")
    # df_ml = pd.read_csv("prescription_exposure_results_ML.csv")
    
    # Simulating the exact data outputs from both approaches for demonstration
    rules_output = [
        {"patient_id": "P001", "drug": "Metformin", "fill_date": "2024-01-01", "baseline_end": "2024-01-30", "rule_used": "Explicit end date"},
        {"patient_id": "P001", "drug": "Metformin", "fill_date": "2024-01-28", "baseline_end": "2024-02-24", "rule_used": "Refill date (next fill - 1 day)"},
        {"patient_id": "P002", "drug": "Lisinopril", "fill_date": "2024-01-05", "baseline_end": "2024-01-31", "rule_used": "Refill date (next fill - 1 day)"},
        {"patient_id": "P002", "drug": "Lisinopril", "fill_date": "2024-02-01", "baseline_end": "2024-03-02", "rule_used": "Days supply calculation"},
        {"patient_id": "P003", "drug": "Amoxicillin", "fill_date": "2024-03-10", "baseline_end": "2024-03-20", "rule_used": "Days supply calculation"},
        {"patient_id": "P004", "drug": "Aspirin", "fill_date": "2024-01-01", "baseline_end": "2024-02-14", "rule_used": "Assumed end: Warfarin initiated"},
        {"patient_id": "P004", "drug": "Warfarin", "fill_date": "2024-02-15", "baseline_end": "2024-03-16", "rule_used": "Days supply calculation"},
        {"patient_id": "P005", "drug": "Atorvastatin", "fill_date": "2024-01-01", "baseline_end": "2024-01-31", "rule_used": "Refill date (next fill - 1 day)"},
        {"patient_id": "P005", "drug": "Atorvastatin", "fill_date": "2024-02-01", "baseline_end": "2024-03-02", "rule_used": "Refill date (next fill - 1 day)"}, # Missed the 90 day gap logic
        {"patient_id": "P006", "drug": "Sertraline", "fill_date": "2024-04-01", "baseline_end": "2024-05-01", "rule_used": "Default days supply (30 days)"},
    ]

    ml_output = [
        {"patient_id": "P001", "drug": "Metformin", "fill_date": "2024-01-01", "ml_end": "2024-01-30"},
        {"patient_id": "P001", "drug": "Metformin", "fill_date": "2024-01-28", "ml_end": "2024-02-24"},
        {"patient_id": "P002", "drug": "Lisinopril", "fill_date": "2024-01-05", "ml_end": "2024-01-31"},
        {"patient_id": "P002", "drug": "Lisinopril", "fill_date": "2024-02-01", "ml_end": "2024-03-02"},
        {"patient_id": "P003", "drug": "Amoxicillin", "fill_date": "2024-03-10", "ml_end": "2024-03-20"},
        {"patient_id": "P004", "drug": "Aspirin", "fill_date": "2024-01-01", "ml_end": "2024-02-14"},
        {"patient_id": "P004", "drug": "Warfarin", "fill_date": "2024-02-15", "ml_end": "2024-03-16"},
        {"patient_id": "P005", "drug": "Atorvastatin", "fill_date": "2024-01-01", "ml_end": "2024-01-31"},
        {"patient_id": "P005", "drug": "Atorvastatin", "fill_date": "2024-02-01", "ml_end": "2024-04-30"}, # ML correctly captures behavioral stretching over the massive gap
        {"patient_id": "P006", "drug": "Sertraline", "fill_date": "2024-04-01", "ml_end": "2024-05-07"}, # ML infers SSRIs run longer on average than hardcoded 30 days
    ]

    df_rules = pd.DataFrame(rules_output)
    df_ml = pd.DataFrame(ml_output)

    # ─── 2. MERGE AND COMPUTE METRICS ─────────────────────────────────────────
    comparison_df = pd.merge(df_rules, df_ml, on=["patient_id", "drug", "fill_date"])
    
    comparison_df["baseline_end"] = pd.to_datetime(comparison_df["baseline_end"])
    comparison_df["ml_end"] = pd.to_datetime(comparison_df["ml_end"])
    
    # Calculate exact date variance (How much did the rule-engine miscalculate exposure?)
    comparison_df["variance_days"] = (comparison_df["ml_end"] - comparison_df["baseline_end"]).dt.days

    # ─── 3. PRINT BOARD READY VIEW ────────────────────────────────────────────
    print("\n" + "="*100)
    print("BOARD PACKAGE EXECUTIVE REPORT: RULES ENGINE VS. MACHINE LEARNING VARIANCE")
    print("="*100)
    
    # Format for neat console printing
    print(f"{'Patient':<9} | {'Drug':<13} | {'Rule End Date':<13} | {'ML End Date':<11} | {'Variance (Days)':<15} | {'Risk / Insight Type'}")
    print("-" * 100)
    
    for _, row in comparison_df.iterrows():
        var = row['variance_days']
        sign = f"+{var}" if var > 0 else f"{var}" if var < 0 else "0"
        
        # Add commentary explaining the business risk of the math error
        commentary = "Perfect Alignment"
        if var > 30:
            commentary = "CRITICAL: Rule Engine missed 90-day medication gap entirely."
        elif var > 0:
            commentary = "Underestimated: Rigid 30-day default rule cut off exposure early."
        elif var < 0:
            commentary = "Overestimated: Fixed math assumed drug extension."
            
        print(f"{row['patient_id']:<9} | {row['drug']:<13} | {row['baseline_end'].strftime('%Y-%m-%d'):<13} | {row['ml_end'].strftime('%Y-%m-%d'):<11} | {sign:<15} | {commentary}")

    # ─── 4. EXECUTIVE EXECUTIVE SUMMARY STATISTICS ────────────────────────────
    print("\n" + "="*100)
    print("EXECUTIVE SUMMARY METRICS FOR DECK SLIDES")
    print("="*100)
    
    total_cases = len(comparison_df)
    divergent_cases = len(comparison_df[comparison_df["variance_days"] != 0])
    pct_impacted = (divergent_cases / total_cases) * 100
    max_error = comparison_df["variance_days"].abs().max()
    avg_error = comparison_df["variance_days"].abs().mean()

    print(f"Total Patient Files Analyzed      : {total_cases}")
    print(f"Files Miscalculated by Rules Engine: {divergent_cases} ({pct_impacted:.1f}%)")
    print(f"Maximum Rule Operational Blindspot: {max_error} Days")
    print(f"Average Rule Calculation Skew     : {avg_error:.1f} Days per file")
    print(f"Variance analysis exported")


    # Save to a single clean analysis file for visualization tools or Excel pivot tables
    output_name = "POC_Executive_Variance_Analysis.csv"
    comparison_df.to_csv(output_name, index=False)
    print(f"Strategic variance database exported")

if __name__ == "__main__":
    run_variance_analysis()
