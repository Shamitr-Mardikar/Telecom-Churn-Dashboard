import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Telecom Churn Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #f0f4ff;
    color: #1a1a2e;
}

h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

section[data-testid="stSidebar"] {
    background: #1a1a2e !important;
    border-right: none;
    padding-top: 2rem;
}
section[data-testid="stSidebar"] * { color: #e0e0ff !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.4rem 0 !important;
}

.main .block-container {
    background: #f0f4ff;
    padding: 2.5rem 3rem;
    max-width: 1300px;
}

.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #ffffff;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-size: 1.05rem;
    color: #ffffff;
    font-weight: 400;
    margin-bottom: 2rem;
    line-height: 1.6;
}

.kpi-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(30,30,80,0.08);
    border-top: 4px solid #4f46e5;
}
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #1a1a2e;
    margin: 0;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #777799;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0.5rem 0 0.3rem;
}
.kpi-note {
    font-size: 0.88rem;
    font-weight: 500;
    color: #444466;
    margin: 0;
}
.kpi-red   { border-top-color: #ef4444; }
.kpi-green { border-top-color: #10b981; }
.kpi-blue  { border-top-color: #4f46e5; }

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #ffffff;
    margin: 2rem 0 0.4rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #dde0f5;
}

.explain-box {
    background: #e8ecff;
    border-left: 4px solid #4f46e5;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    font-size: 0.95rem;
    color: #2a2a4a;
    font-weight: 400;
    line-height: 1.65;
    margin-bottom: 0.8rem;
}
.explain-box strong { color: #1a1a2e; }

.insight-box {
    background: white;
    border-left: 4px solid #10b981;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.95rem;
    color: #1a1a2e;
    font-weight: 400;
    line-height: 1.6;
    margin-bottom: 0.6rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

INDIGO = "#4f46e5"
RED    = "#ef4444"
GREEN  = "#10b981"
AMBER  = "#f59e0b"
SLATE  = "#64748b"

LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(248,249,255,0.8)",
    font=dict(family="DM Sans", color="#2a2a4a", size=13),
    xaxis=dict(gridcolor="#e8ecff", zerolinecolor="#e8ecff", linecolor="#dde0f5", tickfont=dict(size=12)),
    yaxis=dict(gridcolor="#e8ecff", zerolinecolor="#e8ecff", linecolor="#dde0f5", tickfont=dict(size=12)),
    margin=dict(t=50, b=40, l=40, r=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=13)),
    title_font=dict(family="DM Serif Display", size=15, color="#1a1a2e"),
)


@st.cache_data
def load_data():
    df = pd.read_csv("data/Telcom_Dataset.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 10, 20, 30, 40, 50, 60, 72],
        labels=["0–10", "11–20", "21–30", "31–40", "41–50", "51–60", "61+"],
    )
    feats  = df[["tenure", "MonthlyCharges", "TotalCharges"]]
    scaled = StandardScaler().fit_transform(feats)
    km     = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(scaled)

    labels = {}
    for c in range(4):
        sub = df[df["Cluster"] == c]
        cr  = sub["ChurnFlag"].mean()
        ten = sub["tenure"].mean()
        chg = sub["MonthlyCharges"].mean()
        if cr > 0.35:
            labels[c] = "At-Risk"
        elif ten > 40 and chg > 60:
            labels[c] = "Loyal High-Value"
        elif ten < 20:
            labels[c] = "New & Uncertain"
        else:
            labels[c] = "Growth Potential"
    df["Segment"] = df["Cluster"].map(labels)
    return df


try:
    df = load_data()
    ok = True
except FileNotFoundError:
    ok = False

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Telecom Churn")
    st.markdown("---")
    page = st.radio("Navigate to:", [
        "📊 Overview",
        "📋 Contracts & Payments",
        "🔧 Add-On Features",
        "👥 Customer Segments",
    ])

    if ok:
        st.markdown("---")
        st.markdown("**Filter Data**")
        c_opts  = df["Contract"].unique().tolist()
        i_opts  = df["InternetService"].unique().tolist()
        cf   = st.multiselect("Contract Type",    c_opts, default=c_opts)
        ifsv = st.multiselect("Internet Service", i_opts, default=i_opts)
        dff  = df[df["Contract"].isin(cf) & df["InternetService"].isin(ifsv)]
    else:
        dff = None

if not ok:
    st.error("⚠️  **`data/Telcom_Dataset.csv` not found.** Place the CSV inside a `data/` folder next to `app.py`, then restart.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown('<p class="page-title">Overview Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">A high-level summary of the customer base — how many customers we have, how many are leaving, and what the financial impact looks like.</p>', unsafe_allow_html=True)

    total    = len(dff)
    cr       = dff["ChurnFlag"].mean()
    avg_m    = dff["MonthlyCharges"].mean()
    at_risk  = dff[dff["Churn"] == "Yes"]["MonthlyCharges"].sum()

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, f"{total:,}",      "Total Customers",         "in filtered view",            "kpi-blue"),
        (c2, f"{cr:.1%}",        "Churn Rate",              "~1 in 4 customers left",       "kpi-red"),
        (c3, f"${avg_m:.0f}",    "Avg Monthly Charge",      "per customer",                 "kpi-blue"),
        (c4, f"${at_risk:,.0f}", "Monthly Revenue at Risk", "lost from churned customers",  "kpi-red"),
    ]
    for col, val, lbl, note, cls in cards:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
              <p class="kpi-value">{val}</p>
              <p class="kpi-label">{lbl}</p>
              <p class="kpi-note">{note}</p>
            </div>""", unsafe_allow_html=True)

    # Chart 1 — Churn by Tenure
    st.markdown('<p class="section-title">How Long Customers Stay Before Churning</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain-box">
    📖 <strong>What this chart shows:</strong> The X-axis groups customers by how many months they've been with us (called "tenure"). 
    The Y-axis shows the churn rate — the % of customers in each group who left. 
    <br><br>
    <strong>Key insight:</strong> Customers who have been with us for less than 10 months leave at a very high rate (~50%). 
    Once a customer stays past 20 months, they become much more loyal. This tells us: <em>the first few months are make-or-break.</em>
    </div>""", unsafe_allow_html=True)

    tg = dff.groupby("tenure_group", observed=True)["ChurnFlag"].mean().reset_index()
    fig = px.bar(tg, x="tenure_group", y="ChurnFlag",
                 color="ChurnFlag", color_continuous_scale=["#4f46e5", "#ef4444"],
                 labels={"tenure_group": "Tenure Group (months)", "ChurnFlag": "Churn Rate"})
    fig.update_layout(**LAYOUT, coloraxis_showscale=False,
                      title="Churn Rate by Customer Tenure Group")
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    # Chart 2 — Churn by Internet Service
    st.markdown('<p class="section-title">Churn Rate by Internet Service Type</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain-box">
    📖 <strong>What this chart shows:</strong> We compare churn rates across 3 internet service types: DSL, Fiber Optic, and No Internet.
    <br><br>
    <strong>Key insight:</strong> Fiber Optic users churn at ~41% — more than double the DSL rate. 
    This is surprising because Fiber is faster and more expensive. It may suggest service reliability issues or that 
    Fiber customers have higher expectations that aren't being met.
    </div>""", unsafe_allow_html=True)

    ic = dff.groupby("InternetService")["ChurnFlag"].mean().reset_index()
    fig2 = px.bar(ic, x="InternetService", y="ChurnFlag",
                  color_discrete_sequence=[INDIGO],
                  labels={"ChurnFlag": "Churn Rate", "InternetService": "Internet Service"})
    fig2.update_layout(**LAYOUT, title="Churn Rate by Internet Service Type")
    fig2.update_yaxes(tickformat=".0%")
    fig2.update_traces(marker_line_width=0)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">Key Takeaways</p>', unsafe_allow_html=True)
    for txt in [
        f"💸 <strong>${at_risk:,.0f}/month</strong> in revenue is lost to churn. Even recovering 20% = <strong>${at_risk*0.2:,.0f}/month</strong> saved.",
        "🆕 <strong>New customers (0–10 months)</strong> are at highest risk — early engagement programs can make a big difference.",
        "📡 <strong>Fiber Optic</strong> customers churn at the highest rate despite paying the most — a product quality review is needed.",
    ]:
        st.markdown(f'<div class="insight-box">{txt}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — CONTRACTS & PAYMENTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Contracts & Payments":
    st.markdown('<p class="page-title">Contracts & Payment Methods</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Understanding how billing and contract choices relate to whether a customer stays or leaves.</p>', unsafe_allow_html=True)

    # Chart 1
    st.markdown('<p class="section-title">Churn Rate by Contract Type</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain-box">
    📖 <strong>What this chart shows:</strong> We compare three contract types — Month-to-month (no commitment), 
    One-year, and Two-year contracts — and their churn rates.
    <br><br>
    <strong>Key insight:</strong> Month-to-month customers churn at around <strong>42%</strong> while two-year contract 
    holders churn at only about <strong>3%</strong>. This is the single most powerful predictor of churn in the dataset. 
    Encouraging customers to commit to longer contracts is the #1 retention strategy.
    </div>""", unsafe_allow_html=True)

    cc = dff.groupby("Contract")["ChurnFlag"].mean().reset_index().sort_values("ChurnFlag", ascending=False)
    cc["pct"] = (cc["ChurnFlag"] * 100).round(1)
    colors = [RED if c == "Month-to-month" else INDIGO for c in cc["Contract"]]
    fig = px.bar(cc, x="Contract", y="ChurnFlag", text="pct",
                 color="Contract",
                 color_discrete_map={"Month-to-month": RED, "One year": INDIGO, "Two year": GREEN},
                 labels={"ChurnFlag": "Churn Rate", "Contract": "Contract Type"})
    fig.update_traces(texttemplate="%{text}%", textposition="outside", textfont_size=14)
    fig.update_layout(**LAYOUT, showlegend=False, title="Month-to-Month Contracts Have 15× More Churn Than 2-Year Plans")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    # Chart 2
    st.markdown('<p class="section-title">Churn Rate by Payment Method</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain-box">
    📖 <strong>What this chart shows:</strong> This horizontal bar chart shows churn rates broken down by how 
    customers pay — Electronic Check, Mailed Check, Bank Transfer, or Credit Card.
    <br><br>
    <strong>Key insight:</strong> <strong>Electronic Check</strong> users churn at ~45% — far higher than auto-pay 
    methods (Bank Transfer, Credit Card) which are around 15–18%. Customers on auto-pay are less likely to think 
    about their bill each month, which reduces "cancel this" decisions.
    </div>""", unsafe_allow_html=True)

    pc = dff.groupby("PaymentMethod")["ChurnFlag"].mean().reset_index().sort_values("ChurnFlag")
    fig2 = px.bar(pc, x="ChurnFlag", y="PaymentMethod", orientation="h",
                  color="ChurnFlag", color_continuous_scale=[GREEN, RED],
                  labels={"ChurnFlag": "Churn Rate", "PaymentMethod": "Payment Method"})
    fig2.update_layout(**LAYOUT, coloraxis_showscale=False,
                       title="Electronic Check Users Churn Most — Auto-Pay Customers Are Most Loyal")
    fig2.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 — Monthly charges distribution
    st.markdown('<p class="section-title">Monthly Charges: Churned vs Retained Customers</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain-box">
    📖 <strong>What this chart shows:</strong> This is a box plot comparing how much churned vs retained customers 
    pay per month. The box shows where the middle 50% of customers fall; the line in the middle is the median.
    <br><br>
    <strong>Key insight:</strong> Churned customers tend to pay <em>more</em> per month than retained ones. 
    This means high-paying customers are at a higher risk of leaving — possibly because they feel the value 
    doesn't justify the cost.
    </div>""", unsafe_allow_html=True)

    fig3 = px.box(dff, x="Churn", y="MonthlyCharges",
                  color="Churn", color_discrete_map={"No": INDIGO, "Yes": RED},
                  labels={"Churn": "Churned?", "MonthlyCharges": "Monthly Charges ($)"})
    fig3.update_layout(**LAYOUT, showlegend=False,
                       title="Churned Customers Actually Pay More On Average")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<p class="section-title">Key Takeaways</p>', unsafe_allow_html=True)
    for txt in [
        "📋 Converting <strong>month-to-month customers to annual plans</strong> is the single highest-impact retention action available.",
        "💳 Encouraging customers to switch to <strong>auto-pay (bank transfer or credit card)</strong> could significantly reduce churn.",
        "💰 <strong>High-paying customers churn more</strong> — this signals a value perception problem that needs to be addressed.",
    ]:
        st.markdown(f'<div class="insight-box">{txt}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — ADD-ON FEATURES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔧 Add-On Features":
    st.markdown('<p class="page-title">Add-On Feature Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Which optional services actually help retain customers, and which ones don\'t make a difference?</p>', unsafe_allow_html=True)

    services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]

    rows = []
    for svc in services:
        for val in ["Yes", "No"]:
            sub = dff[dff[svc] == val]
            if len(sub):
                rows.append({"Service": svc, "Has Add-On": val,
                              "Churn Rate": sub["ChurnFlag"].mean(),
                              "Count": len(sub)})
    sdf = pd.DataFrame(rows)

    # Chart 1
    st.markdown('<p class="section-title">Does Having an Add-On Reduce Churn?</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain-box">
    📖 <strong>What this chart shows:</strong> For each optional add-on service, we compare the churn rate of 
    customers who have it (green) vs those who don't (red). Each pair of bars represents one service.
    <br><br>
    <strong>Key insight:</strong> <strong>OnlineSecurity</strong> and <strong>TechSupport</strong> are the most 
    effective retention tools — subscribers churn at ~15% vs ~40% for non-subscribers. Streaming services 
    (TV & Movies) show much smaller differences, meaning customers stream but still leave.
    </div>""", unsafe_allow_html=True)

    fig = px.bar(sdf[sdf["Has Add-On"].isin(["Yes", "No"])],
                 x="Service", y="Churn Rate", color="Has Add-On",
                 barmode="group",
                 color_discrete_map={"Yes": GREEN, "No": RED},
                 labels={"Churn Rate": "Churn Rate", "Has Add-On": "Has Add-On?"})
    fig.update_layout(**LAYOUT, title="Security & Support Add-Ons Cut Churn in Half")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    # Chart 2
    st.markdown('<p class="section-title">Add-On Adoption: Churned vs Retained Customers</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain-box">
    📖 <strong>What this chart shows:</strong> This grouped bar chart shows what % of churned (red) vs retained (blue) 
    customers subscribed to each add-on. If retained customers adopt an add-on more, it's a good retention signal.
    <br><br>
    <strong>Key insight:</strong> Retained customers consistently adopt <em>more</em> add-ons across every single 
    service. The biggest gaps are in OnlineSecurity and TechSupport — the same ones that reduce churn most. 
    This confirms that add-ons and loyalty go hand-in-hand.
    </div>""", unsafe_allow_html=True)

    churned_p  = dff[dff["Churn"] == "Yes"][services].apply(lambda x: (x == "Yes").mean())
    retained_p = dff[dff["Churn"] == "No"][services].apply(lambda x: (x == "Yes").mean())

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Churned",  x=services, y=churned_p.values,  marker_color=RED))
    fig2.add_trace(go.Bar(name="Retained", x=services, y=retained_p.values, marker_color=INDIGO))
    fig2.update_layout(**LAYOUT, barmode="group",
                       title="Retained Customers Subscribe to More Add-Ons Across Every Category")
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">Key Takeaways</p>', unsafe_allow_html=True)
    for txt in [
        "🔒 <strong>OnlineSecurity & TechSupport</strong> are the top retention add-ons — ~15% churn vs ~40% without them.",
        "📺 <strong>Streaming services</strong> are popular but don't prevent churn — customers stream and still leave.",
        "🎯 Campaign idea: Offer a discounted <strong>'Protection Bundle'</strong> (Security + TechSupport) to at-risk month-to-month users.",
        "📦 <strong>DeviceProtection & OnlineBackup</strong> have moderate retention value — good as bundle add-ons.",
    ]:
        st.markdown(f'<div class="insight-box">{txt}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — CUSTOMER SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "👥 Customer Segments":
    st.markdown('<p class="page-title">Customer Segments</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Using machine learning (KMeans clustering) to group customers by their behaviour — each group needs a different strategy.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
    📖 <strong>What is KMeans Clustering?</strong> It's a machine learning algorithm that automatically groups 
    customers based on similarities — in this case, how long they've been a customer (tenure), how much they 
    pay per month, and their total spend. We set it to find 4 groups. The algorithm found these groups on its 
    own — we just gave the groups meaningful names based on their characteristics.
    </div>""", unsafe_allow_html=True)

    seg = dff.groupby("Segment").agg(
        Count      =("ChurnFlag", "count"),
        Avg_Tenure =("tenure", "mean"),
        Avg_Monthly=("MonthlyCharges", "mean"),
        Churn_Rate =("ChurnFlag", "mean"),
    ).reset_index().round(1)

    seg_colors = {
        "At-Risk":          RED,
        "Loyal High-Value": AMBER,
        "New & Uncertain":  GREEN,
        "Growth Potential": INDIGO,
    }
    seg_icons = {
        "At-Risk":          "⚠️",
        "Loyal High-Value": "💎",
        "New & Uncertain":  "🆕",
        "Growth Potential": "📈",
    }

    # Segment cards
    st.markdown('<p class="section-title">The 4 Customer Segments</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (_, row) in zip(cols, seg.iterrows()):
        clr  = seg_colors.get(row["Segment"], INDIGO)
        icon = seg_icons.get(row["Segment"], "")
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{clr}">
              <p style="font-size:2rem;margin:0">{icon}</p>
              <p style="font-family:'DM Serif Display',serif;font-size:1rem;color:{clr};font-weight:700;margin:.4rem 0 .8rem">{row['Segment']}</p>
              <p class="kpi-label">Customers</p>
              <p class="kpi-value" style="font-size:1.8rem">{int(row['Count']):,}</p>
              <p class="kpi-label" style="margin-top:.7rem">Avg Tenure</p>
              <p style="font-size:1rem;font-weight:700;color:#1a1a2e;margin:0">{row['Avg_Tenure']:.0f} months</p>
              <p class="kpi-label">Avg Monthly</p>
              <p style="font-size:1rem;font-weight:700;color:#1a1a2e;margin:0">${row['Avg_Monthly']:.0f}</p>
              <p class="kpi-label">Churn Rate</p>
              <p style="font-size:1.1rem;font-weight:700;color:{'#ef4444' if row['Churn_Rate']>0.25 else '#10b981'};margin:0">{row['Churn_Rate']:.1%}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Segment Size Distribution</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explain-box">
        📖 <strong>What this shows:</strong> A pie/donut chart showing what % of our total customer base 
        each segment makes up. This tells us how large each group is so we can prioritize our efforts.
        </div>""", unsafe_allow_html=True)
        fig = px.pie(seg, names="Segment", values="Count", hole=0.45,
                     color="Segment", color_discrete_map=seg_colors)
        fig.update_layout(**LAYOUT, title="How the Customer Base Is Divided")
        fig.update_traces(textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Churn Rate by Segment</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explain-box">
        📖 <strong>What this shows:</strong> A bar chart comparing the churn rate of each segment. 
        The At-Risk group has the highest churn — they need immediate attention.
        </div>""", unsafe_allow_html=True)
        fig2 = px.bar(seg, x="Segment", y="Churn_Rate",
                      color="Segment", color_discrete_map=seg_colors,
                      labels={"Churn_Rate": "Churn Rate", "Segment": ""})
        fig2.update_layout(**LAYOUT, showlegend=False,
                           title="At-Risk Segment Needs Immediate Action")
        fig2.update_yaxes(tickformat=".0%")
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter
    st.markdown('<p class="section-title">Segment Map — Tenure vs Monthly Charges</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain-box">
    📖 <strong>What this scatter plot shows:</strong> Each dot is a customer. The X-axis is how long they've been 
    with us; the Y-axis is how much they pay per month. The color tells us which segment they belong to. 
    <br><br>
    <strong>Key insight:</strong> The segments naturally separate — Loyal High-Value customers cluster in the 
    top-right (long tenure, high spend). At-Risk customers can appear anywhere but often have shorter tenure.
    </div>""", unsafe_allow_html=True)
    fig3 = px.scatter(dff, x="tenure", y="MonthlyCharges", color="Segment",
                      color_discrete_map=seg_colors, opacity=0.5,
                      labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges ($)"})
    fig3.update_layout(**LAYOUT, title="Customer Segments Naturally Separate by Tenure and Spend")
    st.plotly_chart(fig3, use_container_width=True)

    # Playbook
    st.markdown('<p class="section-title">What To Do With Each Segment</p>', unsafe_allow_html=True)
    playbook = [
        ("💎 Loyal High-Value",  AMBER,  "RETAIN & REWARD",   "These are your best customers. Offer loyalty perks, early feature access, and premium support. They're your brand ambassadors."),
        ("📈 Growth Potential",  INDIGO, "UPSELL",            "Offer discounted annual plan upgrades and bundle add-ons (Security + TechSupport) to make them stickier before churn risk grows."),
        ("🆕 New & Uncertain",   GREEN,  "ONBOARD & CONVERT", "The first 6 months are critical. Proactive support, welcome discounts, and addressing friction early can turn them loyal."),
        ("⚠️ At-Risk",           RED,    "WIN-BACK NOW",      "Trigger immediate retention campaigns. Survey for pain points. Offer plan downgrades or targeted discounts to stop the bleed."),
    ]
    for seg_name, clr, action, advice in playbook:
        st.markdown(f"""
        <div class="insight-box" style="border-left-color:{clr}">
          <strong style="color:{clr};font-size:0.85rem;text-transform:uppercase;letter-spacing:0.06em">{action}</strong>
          &nbsp;·&nbsp;<strong>{seg_name}</strong><br>
          <span style="color:#444466">{advice}</span>
        </div>""", unsafe_allow_html=True)