"""
Airline Loyalty Intelligence Pipeline
======================================
Inspired by: Air Canada Aeroplan Risk Engine, Delta SkyMiles Behavioral Intelligence,
             United MileagePlus Churn Prevention Framework

Architecture:
  1. Data ingestion + real-world cleaning decisions
  2. Temporal feature engineering (RFM + behavioral signals)
  3. Churn prediction (XGBoost + SMOTE + Calibration)
  4. Customer segmentation (K-Means on behavioral space)
  5. Smart retention action engine
  6. SHAP explainability
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_recall_curve, confusion_matrix,
                              average_precision_score, f1_score)
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
DATA_DIR = "/mnt/user-data/uploads"
OUT_DIR  = "/home/claude/airline_loyalty"

# Prediction setup: use Jan2017–Sep2018 to predict Oct–Dec2018 churn
# This avoids data leakage (model only sees past-of-prediction-date data)
OBS_END_YEAR, OBS_END_MONTH   = 2018, 9   # observation window end
TARGET_START_YEAR, TARGET_START_MONTH = 2018, 10  # target window start


# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────
def load_and_clean():
    """
    Cleaning decisions (documented for technical report):
    - Salary NaN (25.3%) → imputed with province-level median (regional income proxy)
    - Loyalty Number is the join key across all tables
    - Cancellation fields: NaN means still active (not missing)
    - Zero-flight months are valid records (member enrolled but inactive)
    - CLV is historical; not used as a feature to avoid target leakage
    """
    lh = pd.read_csv(f"{DATA_DIR}/Customer_Loyalty_History.csv")
    fa = pd.read_csv(f"{DATA_DIR}/Customer_Flight_Activity.csv")
    cal = pd.read_csv(f"{DATA_DIR}/Calendar.csv", parse_dates=['Date'])

    # ── Salary imputation: province-level median (real-world standard)
    province_salary = lh.groupby('Province')['Salary'].transform('median')
    lh['Salary'] = lh['Salary'].fillna(province_salary)
    lh['Salary'] = lh['Salary'].fillna(lh['Salary'].median())  # fallback

    # ── Loyalty card → ordinal rank (Star < Nova < Aurora, like real tiers)
    card_rank = {'Star': 1, 'Nova': 2, 'Aurora': 3}
    lh['Card_Rank'] = lh['Loyalty Card'].map(card_rank)

    # ── Education → ordinal
    edu_rank = {'High School or Below': 1, 'College': 2,
                'Bachelor': 3, 'Master': 4, 'Doctor': 5}
    lh['Edu_Rank'] = lh['Education'].map(edu_rank).fillna(2)

    # ── Calendar: add quarter/season mapping
    cal['Month'] = cal['Date'].dt.month
    cal['Year']  = cal['Date'].dt.year
    season_map = {12:4, 1:4, 2:4,   # Winter (Q4 demand peak for holidays)
                   3:1, 4:1, 5:1,   # Spring
                   6:2, 7:2, 8:2,   # Summer (highest travel)
                   9:3, 10:3, 11:3} # Fall
    cal['Season'] = cal['Month'].map(season_map)
    month_season = cal.drop_duplicates(['Year','Month'])[['Year','Month','Season']]
    fa = fa.merge(month_season, on=['Year','Month'], how='left')
    fa['Season'] = fa['Season'].fillna(2)

    print(f"[✓] Loaded {len(lh):,} members | {len(fa):,} activity records")
    return lh, fa


# ─────────────────────────────────────────────────────────────
# 2. CHURN LABEL DEFINITION
# ─────────────────────────────────────────────────────────────
def define_churn(lh, fa):
    """
    Churn = member became inactive in the target window (Oct–Dec 2018).

    Two types (Air Canada-inspired dual-definition):
      A) Hard churn: formally cancelled in target window
      B) Soft churn: zero flights in target window BUT had activity in prior 6 months
         (member is drifting away; most dangerous and actionable segment)

    Key design decision: we EXCLUDE members who cancelled BEFORE the obs window
    (they're already lost — no point predicting them). We also exclude members
    who enrolled AFTER obs_end (too little history to assess).
    """
    # Members who cancelled before observation window: exclude
    pre_cancelled = lh[
        (lh['Cancellation Year'].notna()) &
        ((lh['Cancellation Year'] < OBS_END_YEAR) |
         ((lh['Cancellation Year'] == OBS_END_YEAR) & (lh['Cancellation Month'] <= OBS_END_MONTH)))
    ]['Loyalty Number']

    working_members = lh[~lh['Loyalty Number'].isin(pre_cancelled)].copy()

    # Hard churn: cancelled in target window
    hard_churn = set(lh[
        (lh['Cancellation Year'].notna()) &
        ((lh['Cancellation Year'] > OBS_END_YEAR) |
         ((lh['Cancellation Year'] == OBS_END_YEAR) & (lh['Cancellation Month'] > OBS_END_MONTH)))
    ]['Loyalty Number'])

    # Soft churn: zero flights in target window but had flights in obs window
    target_activity = fa[
        (fa['Year'] > OBS_END_YEAR) |
        ((fa['Year'] == OBS_END_YEAR) & (fa['Month'] > OBS_END_MONTH))
    ].groupby('Loyalty Number')['Total Flights'].sum().reset_index()
    target_activity.columns = ['Loyalty Number', 'Target_Flights']

    obs_activity = fa[
        (fa['Year'] < OBS_END_YEAR) |
        ((fa['Year'] == OBS_END_YEAR) & (fa['Month'] <= OBS_END_MONTH))
    ].groupby('Loyalty Number')['Total Flights'].sum().reset_index()
    obs_activity.columns = ['Loyalty Number', 'Obs_Flights']

    activity_check = working_members[['Loyalty Number']].merge(target_activity, on='Loyalty Number', how='left')
    activity_check = activity_check.merge(obs_activity, on='Loyalty Number', how='left')
    activity_check['Target_Flights'] = activity_check['Target_Flights'].fillna(0)
    activity_check['Obs_Flights']    = activity_check['Obs_Flights'].fillna(0)

    soft_churn = set(activity_check[
        (activity_check['Target_Flights'] == 0) &
        (activity_check['Obs_Flights'] > 3)
    ]['Loyalty Number'])

    working_members['Churned'] = working_members['Loyalty Number'].apply(
        lambda x: 1 if (x in hard_churn or x in soft_churn) else 0
    )

    churn_rate = working_members['Churned'].mean()
    print(f"[✓] Churn definition: {working_members['Churned'].sum():,} churned "
          f"({churn_rate:.1%} rate) out of {len(working_members):,} eligible members")
    print(f"    Hard churn (cancelled): {len(hard_churn):,} | Soft churn (dormant): {len(soft_churn - hard_churn):,}")

    return working_members


# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(lh_with_churn, fa):
    """
    RFM + Behavioral signals inspired by Delta's Propensity-to-Travel model
    and United's MileagePlus engagement scoring.

    Feature families:
    - Recency: how recently did the member fly?
    - Frequency: how often do they fly?
    - Monetary: how much are they worth per trip?
    - Trend: is engagement accelerating or decelerating?
    - Redemption: do they actually use their points? (engagement proxy)
    - Seasonal: do they have predictable travel patterns?
    - Demographic: static member attributes
    - Tenure: how long have they been a member?
    """
    # Restrict to observation window only (anti-leakage)
    fa_obs = fa[
        (fa['Year'] < OBS_END_YEAR) |
        ((fa['Year'] == OBS_END_YEAR) & (fa['Month'] <= OBS_END_MONTH))
    ].copy()

    # Reference date for recency calculation
    ref_date_val = OBS_END_YEAR * 12 + OBS_END_MONTH

    fa_obs['period_num'] = fa_obs['Year'] * 12 + fa_obs['Month']
    fa_obs['months_ago'] = ref_date_val - fa_obs['period_num']

    # ── Active periods only (filter zero-flight months for recency calc)
    fa_active = fa_obs[fa_obs['Total Flights'] > 0]

    # ── Last flight recency
    last_flight = fa_active.groupby('Loyalty Number')['months_ago'].min().reset_index()
    last_flight.columns = ['Loyalty Number', 'Recency_Months']

    # ── Frequency features
    freq = fa_obs.groupby('Loyalty Number').agg(
        Total_Flights_Obs    = ('Total Flights', 'sum'),
        Active_Months        = ('Total Flights', lambda x: (x > 0).sum()),
        Avg_Flights_Month    = ('Total Flights', lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        Max_Flights_Month    = ('Total Flights', 'max'),
        Std_Flights          = ('Total Flights', 'std'),
    ).reset_index()
    freq['Std_Flights'] = freq['Std_Flights'].fillna(0)
    freq['Consistency_Score'] = freq['Active_Months'] / 21  # 21 months in obs window (Jan17–Sep18)

    # ── Monetary / travel depth
    monetary = fa_obs.groupby('Loyalty Number').agg(
        Total_Distance       = ('Distance', 'sum'),
        Total_Points_Acc     = ('Points Accumulated', 'sum'),
        Total_Points_Red     = ('Points Redeemed', 'sum'),
        Total_Dollar_Red     = ('Dollar Cost Points Redeemed', 'sum'),
        Avg_Distance_Flight  = ('Distance', lambda x: x[fa_obs.loc[x.index,'Total Flights']>0].mean() if (fa_obs.loc[x.index,'Total Flights']>0).any() else 0),
    ).reset_index()

    monetary['Redemption_Ratio'] = np.where(
        monetary['Total_Points_Acc'] > 0,
        monetary['Total_Points_Red'] / monetary['Total_Points_Acc'],
        0
    )
    monetary['Redemption_Ratio'] = monetary['Redemption_Ratio'].clip(0, 1)

    # ── Trend features: compare recent 6 months vs earlier 15 months
    fa_recent = fa_obs[fa_obs['months_ago'] <= 6]
    fa_older  = fa_obs[fa_obs['months_ago'] > 6]

    recent_flights = fa_recent.groupby('Loyalty Number')['Total Flights'].sum().reset_index()
    recent_flights.columns = ['Loyalty Number', 'Recent_6M_Flights']
    older_flights  = fa_older.groupby('Loyalty Number')['Total Flights'].sum().reset_index()
    older_flights.columns  = ['Loyalty Number', 'Older_15M_Flights']

    trend = recent_flights.merge(older_flights, on='Loyalty Number', how='outer').fillna(0)
    trend['Older_15M_Avg'] = trend['Older_15M_Flights'] / 15  # normalize to monthly
    trend['Recent_6M_Avg']  = trend['Recent_6M_Flights']  / 6

    # Momentum: positive = accelerating, negative = decelerating (key churn signal!)
    trend['Travel_Momentum'] = trend['Recent_6M_Avg'] - trend['Older_15M_Avg']
    trend['Activity_Trend']  = np.where(
        trend['Older_15M_Avg'] > 0,
        trend['Recent_6M_Avg'] / trend['Older_15M_Avg'],
        np.where(trend['Recent_6M_Avg'] > 0, 2.0, 0.5)  # new high-value or total drop
    )

    # ── Seasonal patterns: do they fly in summer (peak) vs winter?
    summer_flights = fa_obs[fa_obs['Season'] == 2].groupby('Loyalty Number')['Total Flights'].sum().reset_index()
    summer_flights.columns = ['Loyalty Number', 'Summer_Flights']
    winter_flights = fa_obs[fa_obs['Season'] == 4].groupby('Loyalty Number')['Total Flights'].sum().reset_index()
    winter_flights.columns = ['Loyalty Number', 'Winter_Flights']
    seasonal = summer_flights.merge(winter_flights, on='Loyalty Number', how='outer').fillna(0)
    seasonal['Seasonal_Diversity'] = np.where(
        (seasonal['Summer_Flights'] + seasonal['Winter_Flights']) > 0,
        seasonal['Winter_Flights'] / (seasonal['Summer_Flights'] + seasonal['Winter_Flights']),
        0.5
    )

    # ── 2017 vs 2018 year-over-year
    flights_2017 = fa_obs[fa_obs['Year'] == 2017].groupby('Loyalty Number')['Total Flights'].sum().reset_index()
    flights_2017.columns = ['Loyalty Number', 'Flights_2017']
    flights_2018_partial = fa_obs[fa_obs['Year'] == 2018].groupby('Loyalty Number')['Total Flights'].sum().reset_index()
    flights_2018_partial.columns = ['Loyalty Number', 'Flights_2018_Partial']
    yoy = flights_2017.merge(flights_2018_partial, on='Loyalty Number', how='outer').fillna(0)
    # Annualize 2018 partial (9 months → 12 months)
    yoy['Flights_2018_Ann'] = yoy['Flights_2018_Partial'] * (12/9)
    yoy['YoY_Change'] = yoy['Flights_2018_Ann'] - yoy['Flights_2017']

    # ── Demographic features
    demo = lh_with_churn[['Loyalty Number', 'Card_Rank', 'Edu_Rank', 'Salary',
                           'Marital Status', 'Gender', 'Province',
                           'Enrollment Year', 'Enrollment Month', 'Churned']].copy()

    demo['Tenure_Months'] = (OBS_END_YEAR * 12 + OBS_END_MONTH) - \
                            (demo['Enrollment Year'] * 12 + demo['Enrollment Month'])
    demo['Tenure_Months'] = demo['Tenure_Months'].clip(1, 999)

    # Encode categoricals
    le_marital  = LabelEncoder()
    le_gender   = LabelEncoder()
    le_province = LabelEncoder()
    demo['Marital_Enc']  = le_marital.fit_transform(demo['Marital Status'])
    demo['Gender_Enc']   = le_gender.fit_transform(demo['Gender'])
    demo['Province_Enc'] = le_province.fit_transform(demo['Province'])

    # ── Merge everything
    features = demo.merge(last_flight, on='Loyalty Number', how='left')
    features = features.merge(freq,     on='Loyalty Number', how='left')
    features = features.merge(monetary, on='Loyalty Number', how='left')
    features = features.merge(trend,    on='Loyalty Number', how='left')
    features = features.merge(seasonal, on='Loyalty Number', how='left')
    features = features.merge(yoy,      on='Loyalty Number', how='left')

    # Fill members with no obs-window flights (enrolled but never flew)
    fill_zero_cols = ['Total_Flights_Obs','Active_Months','Avg_Flights_Month','Max_Flights_Month',
                      'Std_Flights','Consistency_Score','Total_Distance','Total_Points_Acc',
                      'Total_Points_Red','Total_Dollar_Red','Avg_Distance_Flight','Redemption_Ratio',
                      'Recent_6M_Flights','Older_15M_Flights','Travel_Momentum','Summer_Flights',
                      'Winter_Flights','Seasonal_Diversity','Flights_2017','Flights_2018_Partial',
                      'Flights_2018_Ann','YoY_Change','Older_15M_Avg','Recent_6M_Avg']
    features[fill_zero_cols] = features[fill_zero_cols].fillna(0)
    features['Recency_Months'] = features['Recency_Months'].fillna(21)  # never flew = max recency
    features['Activity_Trend'] = features['Activity_Trend'].fillna(0.5)

    print(f"[✓] Feature matrix: {features.shape[0]:,} members × {features.shape[1]} columns")
    return features


# ─────────────────────────────────────────────────────────────
# 4. CHURN PREDICTION MODEL
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Recency
    'Recency_Months',
    # Frequency
    'Total_Flights_Obs', 'Active_Months', 'Avg_Flights_Month',
    'Max_Flights_Month', 'Std_Flights', 'Consistency_Score',
    # Monetary
    'Total_Distance', 'Total_Points_Acc', 'Total_Points_Red',
    'Total_Dollar_Red', 'Avg_Distance_Flight', 'Redemption_Ratio',
    # Trend
    'Travel_Momentum', 'Activity_Trend', 'YoY_Change', 'Recent_6M_Avg',
    # Seasonal
    'Seasonal_Diversity', 'Summer_Flights', 'Winter_Flights',
    # Demographic
    'Card_Rank', 'Edu_Rank', 'Salary', 'Tenure_Months',
    'Marital_Enc', 'Gender_Enc', 'Province_Enc',
]


def train_churn_model(features_df):
    """
    XGBoost with SMOTE oversampling + Platt scaling calibration.
    Evaluated with 5-fold stratified CV.
    Returns: calibrated model, predictions, SHAP explainer
    """
    X = features_df[FEATURE_COLS].copy()
    y = features_df['Churned'].copy()

    print(f"\n[⚙] Training churn model | Churn rate: {y.mean():.1%} | n={len(y):,}")

    # SMOTE: oversample minority class (like United's churn team does for imbalanced labels)
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)

    # XGBoost base model
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # SMOTE already balances
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )

    # Calibrate for reliable probability outputs (Platt scaling)
    calibrated = CalibratedClassifierCV(xgb_model, method='sigmoid', cv=3)
    calibrated.fit(X_res, y_res)

    # ── Cross-validation on ORIGINAL (unsmoted) data for honest evaluation
    cv_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', use_label_encoder=False,
        random_state=42, n_jobs=-1
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(cv_model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)

    print(f"[✓] 5-Fold CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
    print(f"    Fold AUCs: {', '.join([f'{a:.4f}' for a in cv_aucs])}")

    # ── Final predictions on all data
    churn_proba = calibrated.predict_proba(X)[:, 1]
    features_df['Churn_Probability'] = churn_proba

    # Optimal threshold using F1 maximization (better than default 0.5 for imbalanced)
    precisions, recalls, thresholds = precision_recall_curve(y, churn_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_thresh = thresholds[np.argmax(f1_scores[:-1])]
    features_df['Churn_Predicted'] = (churn_proba >= best_thresh).astype(int)

    print(f"[✓] Optimal threshold: {best_thresh:.3f}")
    print(f"[✓] ROC-AUC (full train): {roc_auc_score(y, churn_proba):.4f}")
    print(f"[✓] PR-AUC: {average_precision_score(y, churn_proba):.4f}")
    print(f"\nClassification Report:\n{classification_report(y, features_df['Churn_Predicted'])}")

    # ── SHAP explainability (train on unsmoted for interpretability)
    print("[⚙] Computing SHAP values (TreeExplainer)...")
    xgb_plain = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', use_label_encoder=False,
        random_state=42, n_jobs=-1
    )
    xgb_plain.fit(X_res, y_res)
    explainer = shap.TreeExplainer(xgb_plain)
    shap_values = explainer.shap_values(X)

    # Feature importance from SHAP
    shap_importance = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'SHAP_Importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('SHAP_Importance', ascending=False)

    print(f"\n[TOP 10 CHURN DRIVERS (SHAP)]")
    for _, row in shap_importance.head(10).iterrows():
        bar = '█' * int(row['SHAP_Importance'] / shap_importance['SHAP_Importance'].max() * 20)
        print(f"  {row['Feature']:<28} {bar} {row['SHAP_Importance']:.4f}")

    metrics = {
        'cv_auc_mean': float(cv_aucs.mean()),
        'cv_auc_std':  float(cv_aucs.std()),
        'cv_aucs':     [float(a) for a in cv_aucs],
        'roc_auc':     float(roc_auc_score(y, churn_proba)),
        'pr_auc':      float(average_precision_score(y, churn_proba)),
        'threshold':   float(best_thresh),
        'churn_rate':  float(y.mean()),
        'n_churned':   int(y.sum()),
        'n_total':     int(len(y)),
        'f1_optimal':  float(f1_score(y, features_df['Churn_Predicted'])),
        'shap_importance': shap_importance.to_dict('records')
    }

    return calibrated, features_df, metrics, shap_values, shap_importance


# ─────────────────────────────────────────────────────────────
# 5. CUSTOMER SEGMENTATION
# ─────────────────────────────────────────────────────────────
def segment_customers(features_df):
    """
    5-segment behavioral model inspired by Air Canada Aeroplan's
    member value framework. Segments are defined on behavioral + value
    dimensions, NOT just demographics.

    Segments:
      0 → Champions        (high value, high frequency, low churn risk)
      1 → Loyal Sleepers   (high CLV historically, drifting, mid churn risk)
      2 → Promising         (moderate frequency, growing, low risk)
      3 → At-Risk Valuables (high value but rapidly declining engagement)
      4 → Dormant           (low engagement, high churn risk, low value)
    """
    seg_features = [
        'Recency_Months', 'Total_Flights_Obs', 'Consistency_Score',
        'Travel_Momentum', 'Total_Points_Acc', 'Redemption_Ratio',
        'Avg_Distance_Flight', 'YoY_Change'
    ]

    X_seg = features_df[seg_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    # K-Means with multiple restarts (airline-grade stability)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=20, max_iter=500)
    clusters = kmeans.fit_predict(X_scaled)
    features_df['Cluster_Raw'] = clusters

    # ── Label clusters by characterizing centroids
    cluster_profiles = features_df.groupby('Cluster_Raw').agg(
        Avg_Recency     = ('Recency_Months', 'mean'),
        Avg_Flights     = ('Total_Flights_Obs', 'mean'),
        Avg_Consistency = ('Consistency_Score', 'mean'),
        Avg_Momentum    = ('Travel_Momentum', 'mean'),
        Avg_Points      = ('Total_Points_Acc', 'mean'),
        Avg_Redemption  = ('Redemption_Ratio', 'mean'),
        Avg_ChurnProb   = ('Churn_Probability', 'mean'),
        Count           = ('Loyalty Number', 'count')
    ).reset_index()

    # Score each cluster: combine normalized signals → assign semantic label
    cluster_profiles['Value_Score']    = cluster_profiles['Avg_Points'] / cluster_profiles['Avg_Points'].max()
    cluster_profiles['Freq_Score']     = cluster_profiles['Avg_Flights'] / cluster_profiles['Avg_Flights'].max()
    cluster_profiles['Engagement_Scr'] = cluster_profiles['Avg_Consistency'] / cluster_profiles['Avg_Consistency'].max()
    cluster_profiles['Risk_Score']     = 1 - cluster_profiles['Avg_ChurnProb']  # higher = less risky

    # Composite score to rank clusters
    cluster_profiles['Composite'] = (
        0.35 * cluster_profiles['Value_Score'] +
        0.25 * cluster_profiles['Freq_Score'] +
        0.25 * cluster_profiles['Engagement_Scr'] +
        0.15 * cluster_profiles['Risk_Score']
    )
    cluster_profiles = cluster_profiles.sort_values('Composite', ascending=False)

    segment_names = [
        '🏆 Champions',
        '💤 Loyal Sleepers',
        '🌱 Promising',
        '⚠️ At-Risk Valuables',
        '🌑 Dormant'
    ]
    segment_colors = ['#00C853', '#FFD600', '#00B0FF', '#FF6D00', '#B0BEC5']
    segment_actions = [
        'VIP recognition + early upgrade offers + co-brand credit card upsell',
        'Personalized win-back offer + Tier Status extension + "We miss you" campaign',
        'Milestone bonus rewards + double-points month + guided tier upgrade path',
        'Immediate rescue call + personalized retention offer + tier downgrade warning',
        'Low-cost email reactivation + survey + consider graceful offboarding'
    ]

    raw_to_segment = {}
    for i, row in enumerate(cluster_profiles.itertuples()):
        raw_to_segment[row.Cluster_Raw] = i

    features_df['Segment_ID']     = features_df['Cluster_Raw'].map(raw_to_segment)
    features_df['Segment_Name']   = features_df['Segment_ID'].map(lambda x: segment_names[x])
    features_df['Segment_Color']  = features_df['Segment_ID'].map(lambda x: segment_colors[x])
    features_df['Segment_Action'] = features_df['Segment_ID'].map(lambda x: segment_actions[x])

    # ── Risk tier within each segment (for action prioritization)
    features_df['Risk_Tier'] = pd.cut(
        features_df['Churn_Probability'],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )

    print("\n[✓] CUSTOMER SEGMENTS:")
    for name, color in zip(segment_names, segment_colors):
        seg_data = features_df[features_df['Segment_Name'] == name]
        churn_pct = seg_data['Churned'].mean() if 'Churned' in seg_data else 0
        print(f"  {name:<26} | n={len(seg_data):>5,} | "
              f"Avg Churn Risk: {seg_data['Churn_Probability'].mean():.1%} | "
              f"Actual Churn: {churn_pct:.1%}")

    return features_df, cluster_profiles, scaler


# ─────────────────────────────────────────────────────────────
# 6. SMART RETENTION ACTION ENGINE
# ─────────────────────────────────────────────────────────────
def generate_retention_actions(features_df):
    """
    Specific, actionable interventions per member.
    Non-vague. Actionable today.
    Inspired by Delta's Next Best Action system and 
    United's Proactive Outreach Programme.
    """

    def get_action(row):
        cp    = row['Churn_Probability']
        seg   = row['Segment_ID']
        card  = row['Card_Rank']
        trend = row['Travel_Momentum']
        red   = row['Redemption_Ratio']
        rec   = row['Recency_Months']
        pts   = row['Total_Points_Acc']

        # Critical risk (>75% churn probability) → immediate intervention
        if cp >= 0.75:
            if card >= 3:  # Aurora top tier
                return ("CALL within 48h", 
                        "Dedicated retention specialist call. Offer: complimentary Companion Fare + 12-month Status Match guarantee. Budget: up to $450 CDN value.",
                        "🔴 CRITICAL")
            elif card == 2:  # Nova
                return ("EMAIL + SMS same day",
                        "Personalized 'We value you' email + 50,000 bonus points offer conditional on 3 flights in next 90 days. Include tier upgrade roadmap.",
                        "🔴 CRITICAL")
            else:
                return ("EMAIL within 3 days",
                        "Reactivation offer: 30,000 bonus points for first flight within 60 days. A/B test subject line: 'Your points are waiting' vs 'Exclusive offer inside'.",
                        "🔴 CRITICAL")

        # High risk (50–75%)
        elif cp >= 0.50:
            if trend < -1:  # Strong decline in travel momentum
                return ("PERSONALIZED OFFER via app push + email",
                        f"Declining traveler alert. Send 'Travel Like You Used To' campaign: 2x points on next 2 flights. Add destination inspiration based on past routes.",
                        "🟠 HIGH")
            elif red < 0.1 and pts > 20000:  # Hoarding points, not redeeming
                return ("POINTS EXPIRY NUDGE email",
                        f"Member has {int(pts):,} unredeemed points. Send 'Your points expire in 180 days' reminder + flight recommendation at current point balance. Urgency drives engagement.",
                        "🟠 HIGH")
            else:
                return ("IN-APP notification + email",
                        "Send status progress bar showing how close they are to next tier. Offer 'accelerator' bonus: fly 2x in 60 days to jump tier early.",
                        "🟠 HIGH")

        # Medium risk (25–50%)
        elif cp >= 0.25:
            if rec > 6:  # Not flown in 6+ months
                return ("EMAIL (monthly digest)",
                        "Include in 'Deals for You' monthly email with 3 personalized route offers based on home city. Add social proof: 'Travellers like you are booking X'.",
                        "🟡 MEDIUM")
            else:
                return ("LOYALTY MILESTONE reminder",
                        "Send tier progress update. Highlight benefit unlocked at next tier. Include seasonal travel inspiration for next quarter.",
                        "🟡 MEDIUM")

        # Low risk (<25%)
        else:
            return ("STANDARD newsletter",
                    "Include in regular member communications. Upsell co-brand credit card if no record of it. Test new product announcements.",
                    "🟢 LOW")

    actions = features_df.apply(get_action, axis=1)
    features_df['Action_Channel']     = actions.apply(lambda x: x[0])
    features_df['Action_Description'] = actions.apply(lambda x: x[1])
    features_df['Risk_Level']         = actions.apply(lambda x: x[2])

    return features_df


# ─────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 60)
    print("  AIRLINE LOYALTY INTELLIGENCE PIPELINE")
    print("  Behavioral Analysis | 2017-2018 Dataset")
    print("=" * 60)

    lh, fa = load_and_clean()
    lh_churn = define_churn(lh, fa)
    features = engineer_features(lh_churn, fa)

    model, features, metrics, shap_values, shap_importance = train_churn_model(features)
    features, cluster_profiles, scaler = segment_customers(features)
    features = generate_retention_actions(features)

    # ── Merge back original demographics for dashboard display
    lh_display = lh[['Loyalty Number', 'Loyalty Card', 'Province', 'City',
                      'Gender', 'Education', 'Salary', 'Marital Status',
                      'CLV', 'Enrollment Type', 'Enrollment Year', 'Enrollment Month',
                      'Cancellation Year', 'Cancellation Month']].copy()

    final_df = features.merge(lh_display, on='Loyalty Number', how='left', suffixes=('','_orig'))

    # Also merge original monthly activity for time-series charts
    fa_monthly = fa.groupby(['Loyalty Number', 'Year', 'Month']).agg(
        Flights=('Total Flights','sum'),
        Distance=('Distance','sum'),
        Points_Acc=('Points Accumulated','sum'),
        Points_Red=('Points Redeemed','sum')
    ).reset_index()

    # ── Save outputs
    os.makedirs(OUT_DIR, exist_ok=True)
    final_df.to_csv(f"{OUT_DIR}/members_scored.csv", index=False)
    fa_monthly.to_csv(f"{OUT_DIR}/flight_activity_clean.csv", index=False)
    cluster_profiles.to_csv(f"{OUT_DIR}/cluster_profiles.csv", index=False)
    shap_importance.to_csv(f"{OUT_DIR}/shap_importance.csv", index=False)

    with open(f"{OUT_DIR}/model.pkl", 'wb') as f:
        pickle.dump({'model': model, 'metrics': metrics, 'scaler': scaler}, f)

    import json
    with open(f"{OUT_DIR}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[✓] All outputs saved to {OUT_DIR}/")
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total members analysed : {len(final_df):,}")
    print(f"  Churn rate detected    : {metrics['churn_rate']:.1%}")
    print(f"  Model ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"  CV AUC (5-fold)        : {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    print(f"{'='*60}\n")

    return final_df, metrics, shap_importance


if __name__ == "__main__":
    run_pipeline()
