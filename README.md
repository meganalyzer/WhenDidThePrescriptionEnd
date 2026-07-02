# When Did the Prescription End?

A comparison of three approaches to one of the most quietly frustrating problems in real-world drug exposure analysis.

---

## The Problem

Think about the last time you picked up a prescription.

You filled it. You came home. Maybe you took it for a few days, felt better, and stopped. Maybe you stretched a 30-day supply across 60 days because life got busy. Maybe you never opened the bottle.

Now imagine trying to analyze drug effectiveness or safety across millions of people like you and me — and the only thing you have is a fill date and a days supply field that someone may or may not have entered correctly.

When did that prescription actually end?

In long-term care facilities, medication orders are managed and closed by clinical staff. The end date exists. For everyone else — people picking up prescriptions at a pharmacy, living their lives — you are almost always inferring it.

And how you infer it changes your cohort, your exposure windows, and potentially your study conclusions.

---

## Three Approaches

### 1. Rules-Based Pipeline (`prescription_exposure.py`)

A deterministic, priority-ordered decision system:

1. Use the explicit end date if recorded
2. Use the next refill date minus one day
3. Calculate from days supply
4. Fall back to assumptions — drug interaction signals, default durations

Clean, auditable, explainable. You can trace exactly why any record got the end date it did.

The limitation: rigid math applied to human behavior creates blind spots. A 90-day gap between refills looks the same as a discontinuation. A patient who stretched their medication gets cut off at Day 30. A new drug not in your interaction dictionary breaks your logic silently.

---

### 2. ML Comparison (`prescription_exposure_ML.py`)

A side-by-side comparison of rule-based outputs versus manually constructed expected ML outputs — designed to illustrate where the two approaches diverge and why.

This file demonstrates the structural cases where rules fail before committing to a full ML implementation.

---

### 3. XGBoost Model (`prescription_exposure_XGBoost.py`)

A real trained model.

**Training strategy:** Use high-confidence rule-based records — where explicit end dates or clean refill data exist — as labeled ground truth. The model learns from cases where we know the answer, then predicts end dates for the ambiguous cases where rules break down.

This is methodologically defensible. You are not asking the model to invent ground truth. You are using your most reliable records to teach it, then applying what it learned to the harder cases.

**Features the model learns from:**
- Days supply
- Drug class
- Fill month (seasonality)
- Whether the drug is chronic or acute
- Historical adherence ratio
- Number of prior fills
- Presence of an interacting drug

**Why XGBoost specifically:** It handles mixed data types and missing values natively, does not require perfectly clean input, and produces feature importance scores — so when it shifts an exposure window, you can see which signals drove that decision. In a clinical context where "the model said so" is not a defensible answer, explainability matters.

---

## Results

| Metric | Result |
|--------|--------|
| Patients in synthetic dataset | 50 |
| Training set | High-confidence records only |
| Overall MAE | 4.0 days |
| High-confidence record MAE | 2.4 days |
| Low-confidence record MAE | 6.7 days |
| Records flagged for human review (>7 day error) | 9.3% |

A 4-day average error on drug exposure windows is meaningful at population scale. The 9.3% of records flagged for review are candidates for exclusion from studies or manual pharmacist audit rather than inclusion with uncertain dates.

---

## The Tradeoff

ML is not a free upgrade.

Even with feature importance, the model is learning from statistical patterns that are real but not always reducible to simple clinical logic. In research and regulatory contexts, that matters.

In practice, records where the model's confidence falls below a threshold often get removed from studies entirely rather than included with uncertain exposure dates. That is a legitimate, defensible choice — but it is worth knowing that ML can increase the number of excluded records rather than reduce overall uncertainty.

The right approach depends on your study design, your regulatory context, and how much ambiguity you can defensibly carry.

---

## Running It

Install dependencies:

```bash
pip install pandas numpy xgboost scikit-learn
```

Mac users — if XGBoost fails on install:

```bash
brew install libomp
```

Run each approach:

```bash
python3 prescription_exposure.py
python3 prescription_exposure_ML.py
python3 prescription_exposure_XGBoost.py
```

---

## Files

| File | Description |
|------|-------------|
| `prescription_exposure.py` | Rule-based pipeline — priority logic for end date estimation |
| `prescription_exposure_ML.py` | Simulated ML comparison — illustrates where rules diverge from probabilistic outputs |
| `prescription_exposure_XGBoost.py` | Real trained XGBoost model with feature engineering and validation |
| `prescription_exposure_results.csv` | Rules-based output with method and flags per record |
| `POC_Executive_Variance_Analysis.csv` | Side-by-side rules vs ML variance |
| `prescription_exposure_XGBoost_results.csv` | Full XGBoost predictions with true vs predicted dates and errors |
| `ML_APPROACH.md` | Detailed methodology and patient scenario walkthrough |

---

## Data

All patient data in this repo is synthetic. No real patient records were used.

---

## Author

**Megha Sharma** 
[LinkedIn](https://linkedin.com/in/meghasharma0892) | [GitHub](https://github.com/meganalyzer)

Building real-world evidence infrastructure for pharma and academic research.
