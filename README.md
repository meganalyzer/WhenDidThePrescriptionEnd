# When Did the Prescription End?

A comparison of two approaches to one of the most quietly frustrating problems in real-world drug exposure analysis.

---

## The Problem

Think about the last time you picked up a prescription.

You filled it. You came home. Maybe you took it for a few days, felt better, and stopped. Maybe you stretched a 30-day supply across 60 days because life got busy. Maybe you never opened the bottle.

Now imagine trying to analyze drug effectiveness or safety across millions of people like you and me — and the only thing you have is a fill date and a days supply field that someone may or may not have entered correctly.

When did that prescription actually end?

In long-term care facilities, medication orders are managed and closed by clinical staff. The end date exists. For everyone else — people picking up prescriptions at a pharmacy, living their lives — you are almost always inferring it.

And how you infer it changes your cohort, your exposure windows, and potentially your study conclusions.

---

## Two Approaches

This repo compares two ways of solving the problem:

### 1. Rules-Based Pipeline (`prescription_exposure.py`)

A deterministic, priority-ordered decision system:

1. Use the explicit end date if recorded
2. Use the next refill date minus one day
3. Calculate from days supply
4. Fall back to assumptions — drug interaction signals, default durations

Clean, auditable, explainable. You can trace exactly why any record got the end date it did.

The limitation: rigid math applied to human behavior creates blind spots. A 90-day gap between refills looks the same as a discontinuation. A patient who stretched their medication gets cut off at Day 30. A new drug not in your interaction dictionary breaks your logic silently.

### 2. ML Comparison (`prescription_exposure_ML.py`)

An XGBoost regressor that learns patterns from the data — drug class, adherence history, fill seasonality — and infers exposure windows probabilistically rather than deterministically.

It handles edge cases the rules engine misses. It adapts to new drugs without manual dictionary updates. It captures behavioral patterns like medication stretching.

---

## The Honest Tradeoff

ML is not a free upgrade.

In research and clinical contexts, precision and explainability are not optional. When a model shifts an exposure window by 59 days, the question is not just whether it is right — it is whether you can prove why it made that decision, and whether a regulator or a reviewer will accept that answer.

Sometimes the honest answer is: we cannot fully explain it. The model learned from patterns that are real but complex, and the math is not simple.

In practice, records where the model's confidence falls below a threshold often get removed from studies entirely rather than included with uncertain exposure dates. That is a legitimate, defensible choice — but it is worth knowing that ML can increase the number of excluded records rather than reduce uncertainty.

The right answer depends on your study design, your regulatory context, and how much ambiguity you can defensibly carry.

---

## What This Repo Shows

| Metric | Result |
|--------|--------|
| Total patient records analyzed | 10 |
| Records where rules and ML disagreed | 2 (20%) |
| Largest exposure window difference | 59 days |
| Average difference | 6.5 days per record |

A 59-day difference in assumed exposure is not a rounding error in a drug safety or comparative effectiveness study. It is a cohort inclusion decision.

---

## Running It

```bash
pip install pandas numpy xgboost
python3 prescription_exposure.py
python3 prescription_exposure_ML.py
```

Outputs:
- `prescription_exposure_results.csv` — rules-based end dates with method and flags
- `POC_Executive_Variance_Analysis.csv` — side-by-side comparison with variance

---

## Data

All patient data in this repo is synthetic. No real patient records were used.

---

## Author

**Megha Sharma** — Clinical Data Engineer, Life Sciences  
[LinkedIn](https://linkedin.com/in/meghasharma0892) | [GitHub](https://github.com/meganalyzer)

Building real-world evidence infrastructure for pharma and academic research.
