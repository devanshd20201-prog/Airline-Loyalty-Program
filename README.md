# SkyGuard™ — Airline Loyalty Behavioral Intelligence Platform

> Award-submission project for **Unlocking Behavioral Intelligence in Airline Loyalty Programs**
> Dataset: 16,700 Canadian loyalty members · 2017–2018 · Air Canada-style program

---

## What This Builds

A two-part system:

1. **ML Pipeline** (`pipeline.py`) — cleans data, engineers 27 behavioral features, trains an XGBoost churn model with SMOTE + Platt calibration, segments members into 5 behavioral groups, and generates specific retention actions per member. Saves all outputs to disk.

2. **Streamlit Dashboard** (`app.py`) — a dark-mode "mission control" UI with 4 pages: Mission Control (KPIs + charts), Segment Explorer (behavioral radar), At-Risk Members (intervention cards + export), and Model Intelligence (SHAP + CV metrics).

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the 4 dataset CSVs in the same folder as pipeline.py:
#    Customer_Loyalty_History.csv
#    Customer_Flight_Activity.csv
#    Calendar.csv
#    Airline_Loyalty_Data_Dictionary.csv

# 3. Run the ML pipeline (trains model, saves all outputs)
python pipeline.py

# 4. Launch the dashboard
streamlit run app.py
```

---

## Project Structure

```
airline_loyalty/
├── pipeline.py              ← ML pipeline (run first)
├── app.py                   ← Streamlit dashboard
├── requirements.txt
├── README.md
│
├── [generated after pipeline.py runs]
├── members_scored.csv       ← All 14,860 members with scores + actions
├── flight_activity_clean.csv
├── cluster_profiles.csv
├── shap_importance.csv
├── metrics.json             ← Model evaluation metrics
└── model.pkl                ← Trained + calibrated model
```

---

## Key Design Decisions

### Churn Definition
No single correct answer exists — we used a **dual-signal definition**:
- **Hard churn**: Formal cancellation recorded in Oct–Dec 2018
- **Soft churn**: Zero flights in Oct–Dec 2018, but ≥3 flights in Jan 2017–Sep 2018 (drifting away without formal exit)

Observation window strictly capped at **Sep 2018** to prevent data leakage.

**Result**: 1,108 churned members (7.5% rate) from 14,860 eligible

### Feature Engineering (27 features, 5 families)
| Family | Features |
|--------|----------|
| Recency | Months since last flight |
| Frequency | Total flights, active months, consistency score, std dev |
| Monetary | Total distance, points accumulated/redeemed, redemption ratio |
| Trend | Travel momentum (recent 6m vs older 15m), YoY change, activity trend |
| Demographic | Card rank, education, salary (province-imputed), tenure, marital status |

### Model
- **XGBoost** (400 trees, depth 5, LR 0.05, subsample 0.8)
- **SMOTE** oversampling (5-NN) to handle 7.5% class imbalance
- **Platt scaling** calibration for reliable probability outputs
- **Threshold**: F1-maximized at 0.162 (not default 0.5)
- **5-fold CV AUC**: 0.6036 ± 0.0093 | **Full ROC-AUC**: 0.7695

### Segmentation (5 behavioral segments)
K-Means (k=5, 20 restarts) on StandardScaled behavioral features. Segments are labeled semantically by composite value + engagement scoring — not just statistically separated.

| Segment | Members | Avg Churn Risk |
|---------|---------|---------------|
| Champions | 688 | 24.0% |
| Loyal Sleepers | 9,450 | 8.0% |
| Promising | 3,435 | 8.7% |
| At-Risk Valuables | 31 | 4.5% |
| Dormant | 1,256 | 5.2% |

### Retention Actions
Non-vague. Each member gets: a **specific channel** (CALL/EMAIL/SMS/IN-APP), a **specific offer** with budget or conditional trigger, and a **suggested timing**. Logic cascades by risk tier × card tier × behavioral signal (momentum, points hoarding, recency).

---

## Model Metrics

```
ROC-AUC (full train) : 0.7695
PR-AUC              : 0.2998
5-Fold CV AUC       : 0.6036 ± 0.0093
F1 (optimal thresh) : 0.2977
Optimal threshold   : 0.162
```

Top churn drivers (SHAP):
1. Active months in observation window
2. Recency of last flight
3. Consistency score
4. Province
5. Points redeemed (engagement proxy)

---

## Inspiration

Real-world systems this draws from:
- **Air Canada Aeroplan** — dual hard/soft churn definition, proactive retention engine
- **Delta SkyMiles Medallion Intelligence** — RFM + behavioral feature engineering, travel momentum signals
- **United MileagePlus Churn Prevention** — SMOTE for imbalanced labels, SHAP explainability for non-technical managers
- **Lufthansa Miles & More** — segment-specific retention budgeting, channel prioritisation by tier

---

## Deliverable Checklist

- [x] Working prototype (Streamlit dashboard, no manual required)
- [x] Churn prediction model with accuracy metrics
- [x] Customer segmentation with behavioral profiles
- [x] Smart retention actions (specific, not vague)
- [x] SHAP explainability per member
- [x] Data leakage prevention (temporal train/test split)
- [x] Documented cleaning decisions (salary imputation, churn definition)
- [x] Export: at-risk member list as CSV
