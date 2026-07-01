# RxExposure — Prescription Exposure End Date Estimator

A real-world evidence (RWE) pipeline for estimating prescription exposure end dates in community-dwelling patients (non-facility).

Built by a clinical data engineer who got tired of seeing this problem handled inconsistently across studies.

---

## The Problem

In facility settings (hospitals, nursing homes), prescription end dates are usually recorded explicitly. A patient is discharged, an order is closed, the date exists.

For community patients picking up prescriptions at a pharmacy — it's messier. The end date is almost never recorded. You have to infer it. And how you infer it materially changes your exposure windows, your cohort counts, and your study conclusions.

Most studies handle this differently. Few document their assumptions clearly. This tool makes the logic explicit, configurable, and auditable.

---

## Priority Logic

End dates are estimated in this order:

| Priority | Source | Notes |
|----------|--------|-------|
| 1 | Explicit prescription end date | Used when recorded — rare in community data |
| 2 | Next refill date | Implies prior fill ended just before refill |
| 3 | Days supply calculation | Fill date + days supply |
| 4 | Assumption-based inference | Drug interaction signals, default assumptions |

---

## Assumption-Based Cases (The Interesting Part)

When none of the above signals exist, the pipeline uses configurable assumptions:

- **Drug interaction signal** — if a conflicting medication is started mid-exposure (e.g. Warfarin initiated while patient is on Aspirin), this may signal discontinuation of the prior drug
- **Refill gap detection** — gaps exceeding days supply + grace period are flagged as possible discontinuation and restart events
- **Low confidence flag** — cases with no signal are estimated using a default days supply and flagged for review

These flags are the findings. They tell you where your exposure windows are uncertain and where sensitivity analyses are needed.

---

## Output

Each prescription fill gets:
- `estimated_end_date` — the best available estimate
- `method` — which priority level was used
- `flag` — any assumptions made or anomalies detected

---

## Configuration

```python
GRACE_PERIOD_DAYS = 7       # Days beyond supply before gap is flagged
DEFAULT_DAYS_SUPPLY = 30    # Fallback when no signal available

INTERACTION_PAIRS = {
    ("Aspirin", "Warfarin"): "Antiplatelet typically discontinued when Anticoagulant initiated",
    ("Metformin", "Insulin"): "Oral hypoglycemic often replaced by Insulin",
}
```

Both are configurable. Drug interaction pairs can be extended for your study context.

---

## Quickstart

```bash
git clone https://github.com/meganalyzer/RxExposure
cd RxExposure
pip install -r requirements.txt
python prescription_exposure.py
```

Results are written to `prescription_exposure_results.csv`.

---

## Synthetic Data

The repo includes `synthetic_prescriptions.json` — a fictional dataset designed to exercise all four priority levels and surface the assumption-based edge cases. No real patient data is used.

---

## Roadmap

- [ ] Fix interaction detection to run across all priority levels
- [ ] Configurable grace periods by drug class (acute vs chronic)
- [ ] Exposure overlap detection — two drugs from same class simultaneously
- [ ] Sensitivity analysis output — show how results change under different assumptions
- [ ] Streamlit UI for non-technical researchers

---

## Why This Matters

Exposure window accuracy directly affects:
- Drug safety studies
- Comparative effectiveness research
- Treatment pattern analyses
- Post-authorization safety studies (PASS)

A 7-day difference in assumed end date can meaningfully change cohort inclusion, outcome timing, and study conclusions. This tool makes those assumptions visible and testable.

---

## Author

**Megha Sharma** — Clinical Data Engineer, Life Sciences  
[LinkedIn](https://linkedin.com/in/meghasharma0892) | [GitHub](https://github.com/meganalyzer)

Building real-world evidence infrastructure for pharma and academic research.

---

## Files

| File | Description |
|------|-------------|
| `prescription_exposure.py` | Rule-based pipeline — priority logic for end date estimation |
| `prescription_exposure_ML.py` | ML comparison — rules engine vs ML output variance analysis |
| `POC_Executive_Variance_Analysis.csv` | Sample output showing where rules fail vs ML |
| `synthetic_prescriptions.json` | Synthetic patient data — no real patient data used |
