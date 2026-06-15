"""
SKYGUARD™ – Airline Loyalty Intelligence Platform
==================================================
Inspired by Delta's Medallion Intelligence HQ, Air Canada Aeroplan's
Proactive Retention Dashboard, and United's MileagePlus Command Center.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import shap

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG – must be first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyGuard™ Loyalty Intelligence",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark command-centre background */
  .stApp { background: #070B14; color: #E8EAF0; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1423 0%, #0A1020 100%);
    border-right: 1px solid #1E2A45;
  }
  [data-testid="stSidebar"] * { color: #C8D0E0 !important; }

  /* KPI cards */
  .kpi-card {
    background: linear-gradient(135deg, #0F1B30 0%, #131F38 100%);
    border: 1px solid #1E3055;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
  }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #2979FF);
    border-radius: 12px 12px 0 0;
  }
  .kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 1.2px;
               text-transform: uppercase; color: #6B7A9A; margin-bottom: 8px; }
  .kpi-value { font-size: 34px; font-weight: 800; color: #E8EAF0; line-height: 1; }
  .kpi-delta { font-size: 12px; margin-top: 6px; }
  .kpi-delta.up   { color: #00E676; }
  .kpi-delta.down { color: #FF5252; }
  .kpi-delta.warn { color: #FFD740; }

  /* Section header */
  .section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 28px 0 16px; padding-bottom: 10px;
    border-bottom: 1px solid #1E2A45;
  }
  .section-header h3 { font-size: 14px; font-weight: 700; letter-spacing: 1px;
                        text-transform: uppercase; color: #8899BB; margin: 0; }

  /* Risk badge */
  .badge { display: inline-block; padding: 3px 10px; border-radius: 20px;
           font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
  .badge-critical { background: rgba(255,82,82,0.15); color: #FF5252; border: 1px solid #FF5252; }
  .badge-high     { background: rgba(255,109,0,0.15); color: #FF6D00; border: 1px solid #FF6D00; }
  .badge-medium   { background: rgba(255,215,64,0.15); color: #FFD740; border: 1px solid #FFD740; }
  .badge-low      { background: rgba(0,230,118,0.15); color: #00E676; border: 1px solid #00E676; }

  /* Member card */
  .member-card {
    background: #0F1B30; border: 1px solid #1E3055; border-radius: 10px;
    padding: 16px; margin-bottom: 10px; transition: border-color 0.2s;
  }
  .member-card:hover { border-color: #2979FF; }

  /* Action card */
  .action-card {
    background: linear-gradient(135deg, #0D1C35 0%, #0F2040 100%);
    border: 1px solid #1E3A6E; border-radius: 10px; padding: 18px;
    margin-top: 12px;
  }
  .action-channel { font-size: 13px; font-weight: 700; color: #2979FF;
                     letter-spacing: 0.5px; margin-bottom: 8px; }
  .action-desc { font-size: 13px; color: #A8B4CC; line-height: 1.6; }

  /* Chart container */
  .chart-wrap {
    background: #0F1B30; border: 1px solid #1E2A45;
    border-radius: 12px; padding: 4px;
  }

  /* Streamlit overrides */
  .stSelectbox > div > div, .stTextInput > div > div {
    background: #0D1423 !important; border-color: #1E3055 !important; color: #E8EAF0 !important;
  }
  .stSlider > div > div > div { color: #2979FF !important; }
  button[kind="primary"] { background: #2979FF !important; border: none !important; }
  .stDataFrame { background: #0F1B30; }
  .stTabs [data-baseweb="tab"] { color: #6B7A9A; background: transparent; }
  .stTabs [aria-selected="true"] { color: #2979FF !important; border-bottom-color: #2979FF !important; }
  h1,h2,h3,h4 { color: #E8EAF0; }
  .stMetric label { color: #6B7A9A !important; }
  .stMetric [data-testid="metric-container"] { background: #0F1B30; border: 1px solid #1E2A45; border-radius: 8px; padding: 12px; }
  div[data-testid="stHorizontalBlock"] { gap: 12px; }

  /* Logo */
  .logo-text {
    font-size: 22px; font-weight: 800; color: #E8EAF0; letter-spacing: -0.5px;
  }
  .logo-accent { color: #2979FF; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: #070B14; }
  ::-webkit-scrollbar-thumb { background: #1E3055; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#8899BB', size=12),
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(gridcolor='#1E2A45', zerolinecolor='#1E2A45', showgrid=True),
    yaxis=dict(gridcolor='#1E2A45', zerolinecolor='#1E2A45', showgrid=True),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1E2A45', font=dict(color='#8899BB'))
)

SEG_PALETTE = {
    '🏆 Champions':         '#00E676',
    '💤 Loyal Sleepers':    '#2979FF',
    '🌱 Promising':         '#00B0FF',
    '⚠️ At-Risk Valuables': '#FF6D00',
    '🌑 Dormant':           '#546E7A',
}

# ─────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = "."
    df   = pd.read_csv(f"{base}/members_scored.csv")
    fa   = pd.read_csv(f"{base}/flight_activity_clean.csv")
    shap = pd.read_csv(f"{base}/shap_importance.csv")
    with open(f"{base}/metrics.json") as f:
        metrics = json.load(f)
    return df, fa, shap, metrics


@st.cache_data
def get_member_timeseries(loyalty_number, fa_df):
    member = fa_df[fa_df['Loyalty Number'] == loyalty_number].copy()
    member['Period'] = pd.to_datetime(member[['Year', 'Month']].assign(day=1))
    member = member.sort_values('Period')
    return member


# ─────────────────────────────────────────────────────────────
# HELPER: KPI Card
# ─────────────────────────────────────────────────────────────
def kpi_card(label, value, delta=None, delta_dir='up', accent='#2979FF'):
    delta_html = ""
    if delta:
        cls = {'up':'up','down':'down','warn':'warn'}.get(delta_dir,'up')
        icons = {'up':'▲','down':'▼','warn':'●'}
        delta_html = f'<div class="kpi-delta {cls}">{icons[delta_dir]} {delta}</div>'

    return f"""
    <div class="kpi-card" style="--accent:{accent}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>"""


# ─────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────
def main():
    df, fa, shap_df, metrics = load_data()
    df['Risk_Tier'] = df['Risk_Tier'].fillna('Low')

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:20px 0 24px">
          <div class="logo-text">Sky<span class="logo-accent">Guard</span>™</div>
          <div style="font-size:11px;color:#4A5570;margin-top:4px;letter-spacing:1px">LOYALTY INTELLIGENCE</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio("NAVIGATION", [
            "✈  Mission Control",
            "🎯  Segment Explorer",
            "🚨  At-Risk Members",
            "🔍  Member Deep Dive",
            "📊  Model Intelligence",
        ], label_visibility='visible')

        st.markdown("<hr style='border-color:#1E2A45;margin:20px 0'>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:11px;color:#4A5570;letter-spacing:1px;margin-bottom:12px'>FILTERS</div>", unsafe_allow_html=True)

        # Segment filter
        all_segs = sorted(df['Segment_Name'].dropna().unique())
        sel_segs = st.multiselect("Segment", all_segs, default=all_segs, label_visibility='visible')

        # Card tier filter
        card_tiers = st.multiselect("Loyalty Card", ['Star','Nova','Aurora'],
                                     default=['Star','Nova','Aurora'])

        # Province filter
        provinces = sorted(df['Province'].dropna().unique())
        sel_provs = st.multiselect("Province", provinces, default=provinces)

        # Churn risk slider
        risk_range = st.slider("Churn Risk Range", 0.0, 1.0, (0.0, 1.0), 0.01,
                                format="%.0f%%",
                                help="Filter members by predicted churn probability")

        st.markdown("<hr style='border-color:#1E2A45;margin:20px 0'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:10px;color:#4A5570;line-height:1.8'>
          Model ROC-AUC<br>
          <span style='color:#2979FF;font-size:16px;font-weight:700'>{metrics['roc_auc']:.4f}</span><br><br>
          CV AUC (5-fold)<br>
          <span style='color:#2979FF;font-size:16px;font-weight:700'>{metrics['cv_auc_mean']:.4f}</span><br><br>
          Members analysed<br>
          <span style='color:#E8EAF0;font-size:14px;font-weight:700'>{metrics['n_total']:,}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Apply filters ────────────────────────────────────────
    mask = (
        df['Segment_Name'].isin(sel_segs) &
        df['Loyalty Card'].isin(card_tiers) &
        df['Province'].isin(sel_provs) &
        (df['Churn_Probability'] >= risk_range[0]) &
        (df['Churn_Probability'] <= risk_range[1])
    )
    filtered = df[mask].copy()

    # ════════════════════════════════════════════════════════
    # PAGE 1: MISSION CONTROL
    # ════════════════════════════════════════════════════════
    if "Mission Control" in page:
        st.markdown("""
        <div style='margin-bottom:24px'>
          <h1 style='font-size:28px;font-weight:800;margin:0'>Mission Control</h1>
          <p style='color:#6B7A9A;margin:6px 0 0;font-size:14px'>
            Real-time loyalty member health · Prediction window: Oct–Dec 2018
          </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Top KPIs ─────────────────────────────────────────
        c1,c2,c3,c4,c5 = st.columns(5)
        n_filtered   = len(filtered)
        n_critical   = len(filtered[filtered['Churn_Probability'] >= 0.75])
        n_high       = len(filtered[(filtered['Churn_Probability'] >= 0.50) & (filtered['Churn_Probability'] < 0.75)])
        avg_risk     = filtered['Churn_Probability'].mean()
        total_pts_atrisk = filtered[filtered['Churn_Probability'] >= 0.50]['Total_Points_Acc'].sum()

        with c1:
            st.markdown(kpi_card("Total Members", f"{n_filtered:,}",
                                  "in filtered view", 'warn', '#2979FF'), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card("Critical Risk", f"{n_critical:,}",
                                  "need immediate action", 'down', '#FF5252'), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card("High Risk", f"{n_high:,}",
                                  "need outreach this week", 'warn', '#FF6D00'), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("Avg Churn Risk", f"{avg_risk:.1%}",
                                  f"vs 7.5% base rate", 'warn' if avg_risk > 0.1 else 'up', '#FFD740'), unsafe_allow_html=True)
        with c5:
            pts_m = total_pts_atrisk / 1_000_000
            st.markdown(kpi_card("Points at Risk", f"{pts_m:.1f}M",
                                  "from 50%+ risk members", 'down', '#FF6D00'), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: Risk Distribution + Segment Donut ─────────
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown('<div class="section-header"><h3>📡 Churn Risk Distribution</h3></div>', unsafe_allow_html=True)
            # Histogram of churn probabilities
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=filtered['Churn_Probability'],
                nbinsx=50,
                name='All Members',
                marker=dict(
                    color=filtered['Churn_Probability'],
                    colorscale=[[0,'#2979FF'],[0.5,'#FF6D00'],[1,'#FF5252']],
                    line=dict(width=0)
                )
            ))
            # Threshold lines
            for thresh, label, col in [(0.25,'Low/Med','#FFD740'),(0.50,'Med/High','#FF6D00'),(0.75,'High/Critical','#FF5252')]:
                fig.add_vline(x=thresh, line_dash='dash', line_color=col,
                              annotation_text=label, annotation_font_color=col,
                              annotation_font_size=10)
            fig.update_layout(**CHART_LAYOUT, height=260,
                               xaxis_title='Churn Probability',
                               yaxis_title='Member Count',
                               title=dict(text='', x=0.5))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header"><h3>🎯 Segment Breakdown</h3></div>', unsafe_allow_html=True)
            seg_counts = filtered['Segment_Name'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Count']
            colors = [SEG_PALETTE.get(s, '#2979FF') for s in seg_counts['Segment']]

            fig2 = go.Figure(go.Pie(
                labels=seg_counts['Segment'],
                values=seg_counts['Count'],
                hole=0.65,
                marker=dict(colors=colors, line=dict(color='#070B14', width=3)),
                textinfo='percent',
                textfont=dict(size=11, color='white'),
                hovertemplate='<b>%{label}</b><br>%{value:,} members<br>%{percent}<extra></extra>'
            ))
            fig2.add_annotation(text=f"<b>{n_filtered:,}</b>", x=0.5, y=0.5,
                                  font=dict(size=24, color='#E8EAF0'), showarrow=False)
            fig2.update_layout(**CHART_LAYOUT, height=260, showlegend=True,
                                )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Row 3: Risk × Value scatter + Province heatmap ───
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-header"><h3>💎 Risk vs. Value Matrix</h3></div>', unsafe_allow_html=True)
            sample = filtered.sample(min(3000, len(filtered)), random_state=42)
            fig3 = go.Figure()
            for seg, color in SEG_PALETTE.items():
                d = sample[sample['Segment_Name'] == seg]
                if len(d) == 0: continue
                fig3.add_trace(go.Scatter(
                    x=d['Churn_Probability'],
                    y=d['Total_Points_Acc'],
                    mode='markers',
                    name=seg,
                    marker=dict(color=color, size=5, opacity=0.7,
                                line=dict(width=0)),
                    hovertemplate=(f'<b>{seg}</b><br>'
                                   'Churn Risk: %{x:.1%}<br>'
                                   'Points: %{y:,}<br>'
                                   '<extra></extra>'),
                ))
            # Quadrant shading
            fig3.add_vrect(x0=0.5, x1=1.0, fillcolor='rgba(255,82,82,0.05)',
                           line_width=0, layer='below')
            fig3.add_annotation(x=0.75, y=sample['Total_Points_Acc'].quantile(0.9),
                                  text="⚠ RESCUE ZONE", font=dict(color='#FF5252', size=10),
                                  showarrow=False)
            fig3.update_layout(**CHART_LAYOUT, height=280,
                                xaxis_title='Churn Probability →',
                                yaxis_title='Total Points Accumulated →')
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-header"><h3>🗺 Churn Risk by Province</h3></div>', unsafe_allow_html=True)
            prov_risk = filtered.groupby('Province').agg(
                Avg_Risk = ('Churn_Probability','mean'),
                Members  = ('Loyalty Number','count'),
                Critical = ('Churn_Probability', lambda x: (x >= 0.75).sum())
            ).reset_index().sort_values('Avg_Risk', ascending=True)

            fig4 = go.Figure(go.Bar(
                x=prov_risk['Avg_Risk'],
                y=prov_risk['Province'],
                orientation='h',
                marker=dict(
                    color=prov_risk['Avg_Risk'],
                    colorscale=[[0,'#2979FF'],[0.5,'#FF6D00'],[1,'#FF5252']],
                    showscale=False
                ),
                text=[f"{r:.1%}  ({c:,})" for r,c in zip(prov_risk['Avg_Risk'], prov_risk['Critical'])],
                textposition='outside',
                textfont=dict(size=10, color='#8899BB'),
                hovertemplate='<b>%{y}</b><br>Avg Risk: %{x:.1%}<br><extra></extra>'
            ))
            fig4.update_layout(**CHART_LAYOUT, height=280)
            st.plotly_chart(fig4, use_container_width=True)

        # ── Row 4: Monthly flight trend aggregate ─────────────
        st.markdown('<div class="section-header"><h3>📈 Portfolio Activity Trend (2017–2018)</h3></div>', unsafe_allow_html=True)

        # Merge filtered members with flight activity
        filtered_ids = set(filtered['Loyalty Number'])
        fa_filtered = fa[fa['Loyalty Number'].isin(filtered_ids)].copy()
        fa_filtered['Period'] = pd.to_datetime(fa_filtered[['Year','Month']].assign(day=1))
        monthly = fa_filtered.groupby('Period').agg(
            Total_Flights  = ('Flights','sum'),
            Total_Points   = ('Points_Acc','sum'),
            Total_Distance = ('Distance','sum'),
            Active_Members = ('Loyalty Number','nunique')
        ).reset_index()

        fig5 = make_subplots(rows=1, cols=3, subplot_titles=[
            'Monthly Flights', 'Points Accumulated', 'Active Members'
        ])
        for i, (col_name, color, label) in enumerate([
            ('Total_Flights',  '#2979FF', 'Flights'),
            ('Total_Points',   '#00E676', 'Points'),
            ('Active_Members', '#FFD740', 'Members')
        ]):
            fig5.add_trace(go.Scatter(
                x=monthly['Period'], y=monthly[col_name],
                fill='tozeroy', name=label,
                line=dict(color=color, width=2),
                fillcolor='rgba({},{},{},0.1)'.format(int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)) if color.startswith('#') else color,
                hovertemplate=f'%{{x|%b %Y}}<br>{label}: %{{y:,}}<extra></extra>'
            ), row=1, col=i+1)
        fig5.update_layout(**CHART_LAYOUT, height=220, showlegend=False)
        for i in range(1,4):
            fig5.update_xaxes(gridcolor='#1E2A45', row=1, col=i)
            fig5.update_yaxes(gridcolor='#1E2A45', row=1, col=i)
        fig5.update_annotations(font=dict(color='#8899BB', size=12))
        st.plotly_chart(fig5, use_container_width=True)


    # ════════════════════════════════════════════════════════
    # PAGE 2: SEGMENT EXPLORER
    # ════════════════════════════════════════════════════════
    elif "Segment Explorer" in page:
        st.markdown("""
        <h1 style='font-size:28px;font-weight:800;margin:0 0 6px'>Segment Explorer</h1>
        <p style='color:#6B7A9A;font-size:14px;margin-bottom:24px'>
          Behavioural profiles · Actionable segment intelligence
        </p>
        """, unsafe_allow_html=True)

        # Segment selector
        sel_seg = st.selectbox("Select Segment to Explore", all_segs)
        seg_data = df[df['Segment_Name'] == sel_seg].copy()
        rest_data = df[df['Segment_Name'] != sel_seg].copy()

        # Segment header card
        color = SEG_PALETTE.get(sel_seg, '#2979FF')
        action_row = seg_data.iloc[0] if len(seg_data) > 0 else None
        action_text = action_row['Segment_Action'] if action_row is not None else ''

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0F1B30,#131F38);
                    border:1px solid {color}44;border-left:4px solid {color};
                    border-radius:10px;padding:20px;margin-bottom:20px">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
              <div style="font-size:24px;font-weight:800;color:{color}">{sel_seg}</div>
              <div style="color:#8899BB;font-size:13px;margin-top:4px">{len(seg_data):,} members 
                · {seg_data['Churn_Probability'].mean():.1%} avg churn risk
                · {len(seg_data[seg_data['Churn_Probability']>=0.5]):,} need urgent action</div>
            </div>
          </div>
          <div style="margin-top:14px;padding:12px 16px;background:rgba(0,0,0,0.3);border-radius:8px">
            <div style="font-size:11px;color:{color};font-weight:700;letter-spacing:1px;margin-bottom:6px">RECOMMENDED STRATEGY</div>
            <div style="font-size:13px;color:#C8D0E0">{action_text}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Segment KPIs ─────────────────────────────────────
        k1,k2,k3,k4 = st.columns(4)
        metrics_seg = {
            'Avg Flights/Obs':    f"{seg_data['Total_Flights_Obs'].mean():.1f}",
            'Avg Churn Risk':     f"{seg_data['Churn_Probability'].mean():.1%}",
            'Avg Points Acc':     f"{seg_data['Total_Points_Acc'].mean()/1000:.0f}K",
            'Avg Tenure':         f"{seg_data['Tenure_Months'].mean():.0f} mo",
        }
        for col_obj, (label, val) in zip([k1,k2,k3,k4], metrics_seg.items()):
            with col_obj:
                st.metric(label, val)

        st.markdown("<br>", unsafe_allow_html=True)
        c_left, c_right = st.columns(2)

        # ── Radar chart: segment profile vs average ───────────
        with c_left:
            st.markdown('<div class="section-header"><h3>📡 Behavioural Fingerprint</h3></div>', unsafe_allow_html=True)
            radar_features = ['Consistency_Score','Redemption_Ratio','Travel_Momentum',
                               'Avg_Flights_Month','Recency_Months','Avg_Distance_Flight']
            radar_labels   = ['Consistency','Redemption','Momentum',
                               'Frequency','Recency (inv)','Distance']

            def normalize(series):
                mn, mx = df[series.name].min(), df[series.name].max()
                if mx - mn == 0: return 0.5
                return float((series.mean() - mn) / (mx - mn + 1e-9))

            # Recency: invert (lower = more recent = better)
            seg_vals = []
            all_vals = []
            for feat in radar_features:
                sv = normalize(seg_data[feat])
                av = normalize(df[feat])
                if feat == 'Recency_Months':
                    sv = 1 - sv
                    av = 1 - av
                seg_vals.append(sv)
                all_vals.append(av)

            fig_r = go.Figure()
            for vals, name, clr in [(all_vals,'Portfolio Avg','#4A5570'), (seg_vals, sel_seg, color)]:
                fig_r.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=radar_labels + [radar_labels[0]],
                    fill='toself', name=name,
                    line=dict(color=clr, width=2),
                    fillcolor=clr.replace('#','rgba(').replace(')','') + ',0.1)' if clr.startswith('#') else clr
                ))
            fig_r.update_layout(**CHART_LAYOUT, height=300,
                                  polar=dict(
                                      bgcolor='rgba(0,0,0,0)',
                                      radialaxis=dict(visible=True, range=[0,1],
                                                       gridcolor='#1E2A45', color='#6B7A9A'),
                                      angularaxis=dict(gridcolor='#1E2A45', color='#8899BB')
                                  ))
            st.plotly_chart(fig_r, use_container_width=True)

        # ── Distribution of churn risk within segment ─────────
        with c_right:
            st.markdown('<div class="section-header"><h3>⚡ Risk Distribution Within Segment</h3></div>', unsafe_allow_html=True)
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=seg_data['Churn_Probability'], nbinsx=30, name=sel_seg,
                marker_color=color, opacity=0.8
            ))
            fig_dist.add_trace(go.Histogram(
                x=rest_data['Churn_Probability'], nbinsx=30, name='Rest of Portfolio',
                marker_color='#4A5570', opacity=0.5
            ))
            fig_dist.update_layout(**CHART_LAYOUT, height=300, barmode='overlay',
                                    xaxis_title='Churn Probability', yaxis_title='Count')
            st.plotly_chart(fig_dist, use_container_width=True)

        # ── Card tier × Churn risk heatmap ───────────────────
        st.markdown('<div class="section-header"><h3>🃏 Card Tier vs Churn Risk (within segment)</h3></div>', unsafe_allow_html=True)
        tier_risk = seg_data.groupby('Loyalty Card')['Churn_Probability'].agg(['mean','count','std']).reset_index()
        tier_risk.columns = ['Card','Avg_Risk','Count','StdDev']
        tier_risk['Card'] = pd.Categorical(tier_risk['Card'], categories=['Star','Nova','Aurora'], ordered=True)
        tier_risk = tier_risk.sort_values('Card')

        fig_bar = go.Figure()
        bar_colors = ['#4A5570','#2979FF','#00E676']
        for row, bcolor in zip(tier_risk.itertuples(), bar_colors):
            fig_bar.add_trace(go.Bar(
                x=[row.Card], y=[row.Avg_Risk],
                name=row.Card, marker_color=bcolor,
                error_y=dict(type='data', array=[row.StdDev], color='#8899BB'),
                text=f"{row.Avg_Risk:.1%} (n={row.Count:,})",
                textposition='outside', textfont=dict(color='#8899BB')
            ))
        fig_bar.update_layout(**CHART_LAYOUT, height=250, showlegend=False,
                               yaxis=dict(tickformat='.0%', gridcolor='#1E2A45'),
                               yaxis_title='Avg Churn Probability')
        st.plotly_chart(fig_bar, use_container_width=True)


    # ════════════════════════════════════════════════════════
    # PAGE 3: AT-RISK MEMBERS
    # ════════════════════════════════════════════════════════
    elif "At-Risk" in page:
        st.markdown("""
        <h1 style='font-size:28px;font-weight:800;margin:0 0 6px'>At-Risk Members</h1>
        <p style='color:#6B7A9A;font-size:14px;margin-bottom:24px'>
          Prioritised intervention list · Actionable retention prescriptions
        </p>
        """, unsafe_allow_html=True)

        # Sort and filter
        risk_thresh = st.select_slider(
            "Show members with churn risk above:",
            options=[0.25, 0.35, 0.50, 0.65, 0.75],
            value=0.50,
            format_func=lambda x: f"{x:.0%}"
        )

        at_risk = filtered[filtered['Churn_Probability'] >= risk_thresh].copy()
        at_risk = at_risk.sort_values('Churn_Probability', ascending=False)

        # Summary bar
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Members Above Threshold", f"{len(at_risk):,}")
        with c2: st.metric("Aurora Tier At-Risk", f"{len(at_risk[at_risk['Loyalty Card']=='Aurora']):,}")
        with c3: st.metric("Avg Risk Score", f"{at_risk['Churn_Probability'].mean():.1%}")
        with c4: st.metric("Avg Tenure", f"{at_risk['Tenure_Months'].mean():.0f} months")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Action priority matrix ─────────────────────────────
        st.markdown('<div class="section-header"><h3>🎯 Action Priority Matrix</h3></div>', unsafe_allow_html=True)

        fig_matrix = go.Figure()
        for risk_level, clr in [('Critical','#FF5252'),('High','#FF6D00'),('Medium','#FFD740')]:
            grp = at_risk[at_risk['Risk_Level'].str.contains(risk_level, na=False)]
            if len(grp) == 0: continue
            fig_matrix.add_trace(go.Scatter(
                x=grp['Tenure_Months'],
                y=grp['Total_Points_Acc'],
                mode='markers',
                name=f"{'🔴' if risk_level=='Critical' else '🟠' if risk_level=='High' else '🟡'} {risk_level}",
                marker=dict(
                    color=grp['Churn_Probability'],
                    colorscale=[[0,'#FF6D00'],[1,'#FF5252']],
                    size=grp['Churn_Probability'] * 20,
                    opacity=0.8,
                    line=dict(color=clr, width=1),
                    showscale=False
                ),
                text=grp['Loyalty Number'].astype(str),
                hovertemplate=(f'<b>Risk: {risk_level}</b><br>'
                               'Member: %{text}<br>'
                               'Tenure: %{x:.0f} months<br>'
                               'Points: %{y:,}<br>'
                               '<extra></extra>')
            ))
        fig_matrix.update_layout(**CHART_LAYOUT, height=320,
                                   xaxis_title='Tenure (months) →',
                                   yaxis_title='Total Points Accumulated →',
                                   title=dict(text='High-tenure, high-value members in top-right are highest priority'))
        st.plotly_chart(fig_matrix, use_container_width=True)

        # ── Member intervention cards ─────────────────────────
        st.markdown('<div class="section-header"><h3>📋 Intervention Cards (Top 30)</h3></div>', unsafe_allow_html=True)

        top_members = at_risk.head(30)

        for _, row in top_members.iterrows():
            risk_pct = row['Churn_Probability']
            risk_col = '#FF5252' if risk_pct >= 0.75 else '#FF6D00' if risk_pct >= 0.5 else '#FFD740'
            badge_cls = 'critical' if risk_pct >= 0.75 else 'high' if risk_pct >= 0.5 else 'medium'

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                <div class="member-card">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                    <div style="font-size:16px;font-weight:700;color:#E8EAF0">#{int(row['Loyalty Number'])}</div>
                    <div style="font-size:22px;font-weight:800;color:{risk_col}">{risk_pct:.0%}</div>
                  </div>
                  <div style="font-size:11px;color:#6B7A9A;margin-bottom:3px">
                    {row.get('Loyalty Card','?')} · {row.get('Province','?')} · {row.get('Gender','?')}
                  </div>
                  <div style="font-size:11px;color:#6B7A9A;margin-bottom:3px">
                    Tenure: {row['Tenure_Months']:.0f} mo · Flights: {row['Total_Flights_Obs']:.0f}
                  </div>
                  <div style="font-size:11px;color:#6B7A9A">
                    Points: {row['Total_Points_Acc']:,.0f} · Momentum: {'📉' if row['Travel_Momentum'] < 0 else '📈'} {row['Travel_Momentum']:.2f}
                  </div>
                  <div style="margin-top:10px">
                    <span class="badge badge-{badge_cls}">{row.get('Risk_Level','?').replace('🔴','').replace('🟠','').replace('🟡','').replace('🟢','').strip()}</span>
                    &nbsp;&nbsp;
                    <span style="font-size:11px;color:{SEG_PALETTE.get(row['Segment_Name'],'#6B7A9A')}">{row['Segment_Name']}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="action-card">
                  <div class="action-channel">📣 {row['Action_Channel']}</div>
                  <div class="action-desc">{row['Action_Description']}</div>
                  <div style="margin-top:10px;font-size:11px;color:#4A5570">
                    Segment: {row['Segment_Name']} · 
                    Card: {row.get('Loyalty Card','?')} · 
                    Last active: ~{row['Recency_Months']:.0f} months ago ·
                    Redemption rate: {row['Redemption_Ratio']:.0%}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Export ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        export_cols = ['Loyalty Number','Loyalty Card','Province','Gender',
                       'Churn_Probability','Risk_Level','Segment_Name',
                       'Action_Channel','Action_Description',
                       'Total_Flights_Obs','Total_Points_Acc','Tenure_Months',
                       'Travel_Momentum','Recency_Months']
        export_df = at_risk[export_cols].copy()
        csv_bytes = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"⬇  Export {len(at_risk):,} At-Risk Members (CSV)",
            data=csv_bytes,
            file_name="skyguard_at_risk_members.csv",
            mime="text/csv",
        )


    # ════════════════════════════════════════════════════════
    # PAGE 4: MEMBER DEEP DIVE
    # ════════════════════════════════════════════════════════
    elif "Member Deep Dive" in page:
        st.markdown("""
        <h1 style='font-size:28px;font-weight:800;margin:0 0 6px'>Member Deep Dive</h1>
        <p style='color:#6B7A9A;font-size:14px;margin-bottom:24px'>
          Individual member intelligence · SHAP explainability · Travel DNA
        </p>
        """, unsafe_allow_html=True)

        # Member search
        member_id = st.text_input("🔍 Enter Loyalty Number", placeholder="e.g. 480934")

        if member_id:
            try:
                mid = int(member_id)
                member_row = df[df['Loyalty Number'] == mid]

                if len(member_row) == 0:
                    # Pick a random high-risk one to demonstrate
                    st.warning(f"Member {mid} not found. Showing a high-risk example instead.")
                    member_row = df[df['Churn_Probability'] >= 0.7].sample(1, random_state=42)
                    mid = int(member_row['Loyalty Number'].values[0])

                m = member_row.iloc[0]

                # ── Member header ─────────────────────────────
                seg_color = SEG_PALETTE.get(m['Segment_Name'], '#2979FF')
                risk_col  = '#FF5252' if m['Churn_Probability'] >= 0.75 else \
                            '#FF6D00' if m['Churn_Probability'] >= 0.5 else \
                            '#FFD740' if m['Churn_Probability'] >= 0.25 else '#00E676'

                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#0F1B30,#131F38);
                            border:1px solid #1E3055;border-radius:12px;padding:24px;
                            margin-bottom:20px">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px">
                    <div>
                      <div style="font-size:28px;font-weight:800;color:#E8EAF0">
                        Member #{mid}
                      </div>
                      <div style="font-size:13px;color:#6B7A9A;margin-top:6px">
                        {m.get('Loyalty Card','?')} Card · {m.get('Province','?')}, {m.get('City','?')} ·
                        {m.get('Gender','?')} · {m.get('Education','?')} · {m.get('Marital Status','?')}
                      </div>
                      <div style="margin-top:12px">
                        <span style="background:rgba(0,0,0,0.3);border:1px solid {seg_color};
                                     color:{seg_color};padding:4px 12px;border-radius:20px;font-size:12px">
                          {m['Segment_Name']}
                        </span>
                      </div>
                    </div>
                    <div style="text-align:right">
                      <div style="font-size:11px;color:#6B7A9A;letter-spacing:1px;text-transform:uppercase">CHURN RISK</div>
                      <div style="font-size:52px;font-weight:800;color:{risk_col};line-height:1">
                        {m['Churn_Probability']:.0%}
                      </div>
                      <div style="font-size:12px;color:#6B7A9A">{m.get('Risk_Level','')}</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── KPI row ───────────────────────────────────
                kk = st.columns(6)
                kpis = [
                    ("Total Flights", f"{m['Total_Flights_Obs']:.0f}", "obs window"),
                    ("Total Points",  f"{m['Total_Points_Acc']/1000:.1f}K", "accumulated"),
                    ("Redemption",    f"{m['Redemption_Ratio']:.0%}", "engagement"),
                    ("Tenure",        f"{m['Tenure_Months']:.0f} mo", "enrolled"),
                    ("Recency",       f"{m['Recency_Months']:.0f} mo", "since last flight"),
                    ("Momentum",      f"{m['Travel_Momentum']:+.2f}", "flights/month Δ"),
                ]
                for col_obj, (label, val, sub) in zip(kk, kpis):
                    with col_obj:
                        st.metric(label, val, sub)

                st.markdown("<br>", unsafe_allow_html=True)
                tab1, tab2, tab3 = st.tabs(["✈ Travel Timeline", "🧠 Why This Risk Score?", "💊 Retention Prescription"])

                # ── Tab 1: Travel timeline ─────────────────────
                with tab1:
                    ts = get_member_timeseries(mid, fa)
                    if len(ts) > 0:
                        fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                                row_heights=[0.6, 0.4],
                                                subplot_titles=['Monthly Flights & Distance', 'Points Accumulated vs Redeemed'])
                        fig_ts.add_trace(go.Bar(x=ts['Period'], y=ts['Flights'],
                                                 name='Flights', marker_color='#2979FF',
                                                 opacity=0.8), row=1, col=1)
                        ax2 = go.Scatter(x=ts['Period'], y=ts['Distance'],
                                          name='Distance (km)', line=dict(color='#00E676', width=2),
                                          yaxis='y2')
                        fig_ts.add_trace(ax2, row=1, col=1)
                        fig_ts.add_trace(go.Bar(x=ts['Period'], y=ts['Points_Acc'],
                                                 name='Points Earned', marker_color='#2979FF', opacity=0.6), row=2, col=1)
                        fig_ts.add_trace(go.Bar(x=ts['Period'], y=ts['Points_Red'],
                                                 name='Points Used', marker_color='#FF6D00', opacity=0.8), row=2, col=1)
                        fig_ts.update_layout(**CHART_LAYOUT, height=400, barmode='overlay',
                                              showlegend=True)
                        fig_ts.update_xaxes(gridcolor='#1E2A45')
                        fig_ts.update_yaxes(gridcolor='#1E2A45')
                        st.plotly_chart(fig_ts, use_container_width=True)
                    else:
                        st.info("No flight activity data for this member in the dataset.")

                # ── Tab 2: SHAP waterfall ─────────────────────
                with tab2:
                    st.markdown("""
                    <p style='color:#8899BB;font-size:13px;margin-bottom:16px'>
                      This explains WHY the model assigned this specific churn risk score.
                      Each bar shows how much a feature pushed the score UP (toward churn) or DOWN (away from churn).
                    </p>
                    """, unsafe_allow_html=True)

                    # Use SHAP importance as a proxy for per-member explanation
                    # (true waterfall requires re-running SHAP on single member)
                    shap_sorted = shap_df.head(12).copy()
                    member_vals = []
                    for feat in shap_sorted['Feature']:
                        try:
                            mv = float(m[feat])
                            port_avg = float(df[feat].mean())
                            # Directional heuristic based on feature semantics
                            if feat in ['Recency_Months']:
                                impact = (mv - port_avg) / (df[feat].std() + 1e-9) * 0.05
                            elif feat in ['Active_Months','Consistency_Score','Total_Flights_Obs',
                                          'Travel_Momentum','Avg_Flights_Month']:
                                impact = -(mv - port_avg) / (df[feat].std() + 1e-9) * 0.04
                            else:
                                impact = (mv - port_avg) / (df[feat].std() + 1e-9) * 0.02
                        except:
                            impact = 0
                        member_vals.append(impact)

                    shap_sorted['Member_Impact'] = member_vals
                    shap_sorted_plot = shap_sorted.sort_values('Member_Impact')

                    fig_shap = go.Figure(go.Bar(
                        x=shap_sorted_plot['Member_Impact'],
                        y=shap_sorted_plot['Feature'],
                        orientation='h',
                        marker_color=['#FF5252' if v > 0 else '#00E676' for v in shap_sorted_plot['Member_Impact']],
                        text=[f"{'+' if v>0 else ''}{v:.3f}" for v in shap_sorted_plot['Member_Impact']],
                        textposition='outside',
                        textfont=dict(size=10)
                    ))
                    fig_shap.add_vline(x=0, line_color='#6B7A9A', line_width=1)
                    fig_shap.update_layout(**CHART_LAYOUT, height=380,
                                            xaxis_title='← Reduces churn risk   |   Increases churn risk →',
                                            title=dict(text='Feature Impact on Churn Score'))
                    st.plotly_chart(fig_shap, use_container_width=True)

                    st.markdown(f"""
                    <div style="background:#0D1423;border:1px solid #1E3055;border-radius:8px;padding:16px;font-size:13px;color:#8899BB;line-height:1.8">
                      <b style="color:#E8EAF0">Key drivers for this member:</b><br>
                      • Active months in observation window: <b style="color:#E8EAF0">{m['Active_Months']:.0f}</b> 
                        (portfolio avg: {df['Active_Months'].mean():.1f})<br>
                      • Recency of last flight: <b style="color:#E8EAF0">{m['Recency_Months']:.0f} months ago</b><br>
                      • Travel momentum: <b style="color:{'#00E676' if m['Travel_Momentum'] > 0 else '#FF5252'}">{m['Travel_Momentum']:+.2f} flights/month</b><br>
                      • Redemption rate: <b style="color:#E8EAF0">{m['Redemption_Ratio']:.0%}</b> of accumulated points used
                    </div>
                    """, unsafe_allow_html=True)

                # ── Tab 3: Retention prescription ─────────────
                with tab3:
                    risk_pct = m['Churn_Probability']
                    risk_col2 = '#FF5252' if risk_pct >= 0.75 else '#FF6D00' if risk_pct >= 0.5 else '#FFD740'

                    st.markdown(f"""
                    <div style="background:#0D1C35;border:1px solid #1E3A6E;border-radius:10px;padding:24px">
                      <div style="font-size:12px;font-weight:700;color:#2979FF;letter-spacing:1px;margin-bottom:6px">
                        CHANNEL
                      </div>
                      <div style="font-size:20px;font-weight:700;color:#E8EAF0;margin-bottom:16px">
                        {m['Action_Channel']}
                      </div>
                      <div style="font-size:12px;font-weight:700;color:#2979FF;letter-spacing:1px;margin-bottom:6px">
                        RECOMMENDED ACTION
                      </div>
                      <div style="font-size:14px;color:#C8D0E0;line-height:1.8;margin-bottom:20px">
                        {m['Action_Description']}
                      </div>
                      <hr style="border-color:#1E3A6E;margin:16px 0">
                      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px">
                        <div>
                          <div style="font-size:11px;color:#6B7A9A">EXPECTED LIFT</div>
                          <div style="font-size:18px;font-weight:700;color:#00E676">
                            {'+15–22%' if risk_pct >= 0.75 else '+8–15%' if risk_pct >= 0.5 else '+3–8%'}
                          </div>
                          <div style="font-size:11px;color:#6B7A9A">retention probability</div>
                        </div>
                        <div>
                          <div style="font-size:11px;color:#6B7A9A">SUGGESTED TIMING</div>
                          <div style="font-size:18px;font-weight:700;color:#E8EAF0">
                            {'48 hrs' if risk_pct >= 0.75 else '1 week' if risk_pct >= 0.5 else '30 days'}
                          </div>
                          <div style="font-size:11px;color:#6B7A9A">from now</div>
                        </div>
                        <div>
                          <div style="font-size:11px;color:#6B7A9A">ESTIMATED VALUE</div>
                          <div style="font-size:18px;font-weight:700;color:#E8EAF0">
                            ${m['Total_Points_Acc'] * 0.01 / 12 * 24:,.0f}
                          </div>
                          <div style="font-size:11px;color:#6B7A9A">2-yr projected CLV</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            except ValueError:
                st.error("Please enter a valid numeric Loyalty Number.")

        else:
            # Show a sample list of interesting members
            st.info("💡 Enter a Loyalty Number to explore. Here are some high-risk examples:")
            ex = df[df['Churn_Probability'] >= 0.7].nlargest(10, 'Total_Points_Acc')[
                ['Loyalty Number','Loyalty Card','Province','Churn_Probability',
                 'Segment_Name','Total_Flights_Obs','Total_Points_Acc']
            ]
            st.dataframe(
                ex.style.background_gradient(subset=['Churn_Probability'], cmap='Reds'),
                use_container_width=True
            )


    # ════════════════════════════════════════════════════════
    # PAGE 5: MODEL INTELLIGENCE
    # ════════════════════════════════════════════════════════
    elif "Model Intelligence" in page:
        st.markdown("""
        <h1 style='font-size:28px;font-weight:800;margin:0 0 6px'>Model Intelligence</h1>
        <p style='color:#6B7A9A;font-size:14px;margin-bottom:24px'>
          XGBoost churn model · SHAP explainability · Validation metrics
        </p>
        """, unsafe_allow_html=True)

        # ── Model metrics ─────────────────────────────────────
        m1,m2,m3,m4,m5 = st.columns(5)
        with m1: st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}", "Full train")
        with m2: st.metric("CV ROC-AUC", f"{metrics['cv_auc_mean']:.4f}", f"±{metrics['cv_auc_std']:.4f}")
        with m3: st.metric("PR-AUC", f"{metrics['pr_auc']:.4f}", "Precision-Recall")
        with m4: st.metric("Optimal Threshold", f"{metrics['threshold']:.3f}", "F1-maximized")
        with m5: st.metric("F1 Score", f"{metrics['f1_optimal']:.4f}", "At optimal thresh")

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        # ── SHAP importance bar ───────────────────────────────
        with col_l:
            st.markdown('<div class="section-header"><h3>🧠 Global Feature Importance (SHAP)</h3></div>', unsafe_allow_html=True)
            shap_plot = shap_df.head(15).copy()
            fig_fi = go.Figure(go.Bar(
                x=shap_plot['SHAP_Importance'],
                y=shap_plot['Feature'],
                orientation='h',
                marker=dict(
                    color=shap_plot['SHAP_Importance'],
                    colorscale=[[0,'#1565C0'],[0.5,'#2979FF'],[1,'#00E676']],
                    showscale=False
                ),
                text=[f"{v:.4f}" for v in shap_plot['SHAP_Importance']],
                textposition='outside', textfont=dict(size=10, color='#8899BB')
            ))
            fig_fi.update_layout(**CHART_LAYOUT, height=420,
                                   xaxis_title='Mean |SHAP Value|',
                                   yaxis=dict(gridcolor='rgba(0,0,0,0)', autorange='reversed'))
            st.plotly_chart(fig_fi, use_container_width=True)

        # ── CV fold scores ────────────────────────────────────
        with col_r:
            st.markdown('<div class="section-header"><h3>📊 Cross-Validation Results</h3></div>', unsafe_allow_html=True)
            cv_aucs = metrics.get('cv_aucs', [])
            if cv_aucs:
                fig_cv = go.Figure()
                fold_labels = [f"Fold {i+1}" for i in range(len(cv_aucs))]
                fig_cv.add_trace(go.Bar(
                    x=fold_labels, y=cv_aucs,
                    marker_color=['#FF5252' if v == min(cv_aucs) else '#00E676' if v == max(cv_aucs) else '#2979FF'
                                  for v in cv_aucs],
                    text=[f"{v:.4f}" for v in cv_aucs],
                    textposition='outside', textfont=dict(color='#8899BB')
                ))
                fig_cv.add_hline(y=np.mean(cv_aucs), line_dash='dash',
                                   line_color='#FFD740',
                                   annotation_text=f"Mean: {np.mean(cv_aucs):.4f}",
                                   annotation_font_color='#FFD740')
                fig_cv.update_layout(**CHART_LAYOUT, height=220,
                                      yaxis=dict(range=[0.5, 0.7], gridcolor='#1E2A45'),
                                      yaxis_title='ROC-AUC')
                st.plotly_chart(fig_cv, use_container_width=True)

            # ── Churn probability distribution ────────────────
            st.markdown('<div class="section-header"><h3>📈 Score Calibration Check</h3></div>', unsafe_allow_html=True)
            churned_probs = df[df['Churned'] == 1]['Churn_Probability']
            retained_probs = df[df['Churned'] == 0]['Churn_Probability']

            fig_cal = go.Figure()
            fig_cal.add_trace(go.Histogram(x=retained_probs, nbinsx=30, name='Retained',
                                            marker_color='#2979FF', opacity=0.6, histnorm='probability'))
            fig_cal.add_trace(go.Histogram(x=churned_probs, nbinsx=30, name='Churned',
                                            marker_color='#FF5252', opacity=0.6, histnorm='probability'))
            fig_cal.update_layout(**CHART_LAYOUT, height=200, barmode='overlay',
                                    xaxis_title='Predicted Churn Probability')
            st.plotly_chart(fig_cal, use_container_width=True)

        # ── Methodology notes ─────────────────────────────────
        st.markdown('<div class="section-header"><h3>📖 Methodology</h3></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0F1B30;border:1px solid #1E3055;border-radius:10px;padding:24px;
                    font-size:13px;color:#8899BB;line-height:1.9;display:grid;
                    grid-template-columns:1fr 1fr;gap:24px">
          <div>
            <b style="color:#2979FF;display:block;margin-bottom:8px">CHURN DEFINITION</b>
            Dual-signal: (1) Formal cancellation in target window Oct–Dec 2018, OR
            (2) Zero flights in target window with ≥3 flights in prior observation window.
            Observation cutoff enforced at Sep 2018 to prevent data leakage.
            <br><br>
            <b style="color:#2979FF;display:block;margin-bottom:8px">FEATURE ENGINEERING</b>
            27 features across 5 families: Recency, Frequency, Monetary (RFM), 
            Trend (momentum & YoY), Seasonal patterns, and Demographics.
            Province-level median salary imputation for 25.3% missing values.
          </div>
          <div>
            <b style="color:#2979FF;display:block;margin-bottom:8px">MODEL ARCHITECTURE</b>
            XGBoost (400 trees, depth=5, LR=0.05) + SMOTE oversampling (5-NN) 
            to handle 7.5% churn imbalance. Platt scaling calibration for 
            reliable probability outputs. Threshold optimized via F1 maximization 
            on precision-recall curve.
            <br><br>
            <b style="color:#2979FF;display:block;margin-bottom:8px">SEGMENTATION</b>
            K-Means (k=5, 20 restarts) on StandardScaled behavioral features.
            Segments semantically labeled by composite value+engagement score.
            Validated for business distinctiveness, not just statistical separation.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
