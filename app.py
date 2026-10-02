import numpy as np, pandas as pd, streamlit as st, plotly.express as px, plotly.graph_objects as go
from scipy import stats
from utils import *

st.set_page_config(page_title="6G Network vs Manufacturing Efficiency", page_icon="📡", layout="wide")
DATA_PATH = "Thales_Group_Manufacturing.csv"
COLORS = {"Low": "#E4572E", "Medium": "#F3A712", "High": "#2E9E6B"}

@st.cache_data(show_spinner="Loading dataset...")
def get_data(path):
    return mark_unstable(load_data(path))

try:
    df = get_data(DATA_PATH)
except FileNotFoundError:
    up = st.sidebar.file_uploader("Upload Thales_Group_Manufacturing.csv", type="csv")
    if up is None:
        st.info("Place `Thales_Group_Manufacturing.csv` next to app.py or upload it in the sidebar."); st.stop()
    df = mark_unstable(load_data(up))

# ---------------- Sidebar filters ----------------
st.sidebar.title("🎛️ Filters")
nq = st.sidebar.multiselect("Network quality", ["Low", "Medium", "High"], default=["Low", "Medium", "High"],
                            help="Tertiles of the Network Stability Index (NSI).")
eff = st.sidebar.multiselect("Efficiency class", EFF_ORDER, default=EFF_ORDER)
modes = ["All"] + sorted(df["Operation_Mode"].unique())
mode = st.sidebar.selectbox("Operation mode", modes)
dmin, dmax = df["Datetime"].dt.date.min(), df["Datetime"].dt.date.max()
dr = st.sidebar.date_input("Time window (dates)", (dmin, dmax), min_value=dmin, max_value=dmax)
hrs = st.sidebar.slider("Hour of day", 0, 23, (0, 23))

f = df.copy()
f = f[f["Network_Quality"].isin(nq) & f["Efficiency_Status"].isin(eff)]
if mode != "All": f = f[f["Operation_Mode"] == mode]
if isinstance(dr, tuple) and len(dr) == 2:
    f = f[(f["Date"] >= dr[0]) & (f["Date"] <= dr[1])]
f = f[f["Hour"].between(*hrs)]
st.sidebar.metric("Records in view", f"{len(f):,}", f"{len(f)/len(df):.0%} of dataset")
if len(f) < 500:
    st.warning("Too few records for the current filters - widen the selection."); st.stop()

st.title("📡 Impact of 6G Network Performance on Manufacturing Efficiency")
st.caption("Thales Group · Unified Mentor project — a connectivity-first view of smart-factory efficiency")

# ---------------- KPIs ----------------
sens = latency_sensitivity(f); plr = packet_loss_impact_ratio(f); bp = efficiency_breakpoint(f)
c = st.columns(5)
c[0].metric("Network Stability Index", f"{f['NSI'].mean():.1f} / 100", help="100 - 50%*latency/50ms - 50%*loss/5%, scaled to 0-100")
c[1].metric("Latency Sensitivity Score", f"{sens['eff_per_10ms']:+.4f}", help="Change in efficiency score (Low=0, Med=1, High=2) per +10 ms. p=%.3f" % sens["eff_p"])
c[2].metric("Packet-Loss Impact Ratio", f"{plr:.2%}", help="Production-speed drop: loss >4% vs loss <=1%")
c[3].metric("Network-Efficiency Breakpoint", f"{bp[0]:.1f} ms" if bp[0] == bp[0] else "n/a", help="Latency where share of Medium+High efficiency falls the most; check significance in tab 4")
c[4].metric("High+Medium efficiency", f"{(f['Efficiency_Score']>=1).mean():.1%}")

t1, t2, t3, t4, t5 = st.tabs(["1️⃣ Network Overview", "2️⃣ Network vs Efficiency", "3️⃣ Quality & Error Impact",
                              "4️⃣ 6G Optimization Insights", "ℹ️ Method & Data"])

# ---------------- Tab 1 ----------------
with t1:
    st.subheader("Latency & packet-loss trends")
    grain = st.radio("Aggregation", ["Hourly", "Daily"], horizontal=True)
    rule = "h" if grain == "Hourly" else "D"
    ts = f.set_index("Datetime")[["Network_Latency_ms", "Packet_Loss_%", "NSI"]].resample(rule).mean().dropna()
    fig = go.Figure()
    fig.add_scatter(x=ts.index, y=ts["Network_Latency_ms"], name="Latency (ms)", line=dict(color="#1f77b4"))
    fig.add_scatter(x=ts.index, y=ts["Packet_Loss_%"], name="Packet loss (%)", yaxis="y2", line=dict(color="#d62728"))
    fig.update_layout(yaxis=dict(title="Latency (ms)"), yaxis2=dict(title="Packet loss (%)", overlaying="y", side="right"),
                      height=380, legend=dict(orientation="h", y=1.1), margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(f, x="Network_Latency_ms", nbins=50, title="Latency distribution", color_discrete_sequence=["#1f77b4"]), use_container_width=True)
    b.plotly_chart(px.histogram(f, x="Packet_Loss_%", nbins=50, title="Packet-loss distribution", color_discrete_sequence=["#d62728"]), use_container_width=True)
    st.subheader("Network stability scorecards")
    fs = mark_unstable(f)
    sc = st.columns(4)
    sc[0].metric("Mean latency", f"{f['Network_Latency_ms'].mean():.1f} ms", f"p95 {f['Network_Latency_ms'].quantile(.95):.1f}")
    sc[1].metric("Mean packet loss", f"{f['Packet_Loss_%'].mean():.2f} %", f"p95 {f['Packet_Loss_%'].quantile(.95):.2f}")
    sc[2].metric("Latency jitter (σ)", f"{f['Network_Latency_ms'].std():.1f} ms")
    sc[3].metric("Unstable-hour share", f"{(fs['Period_Stability']=='Unstable').mean():.0%}", help="Hours whose mean NSI is in the bottom 25% of all hours")
    q = f["Network_Quality"].value_counts().reindex(["Low", "Medium", "High"]).reset_index()
    q.columns = ["Network quality", "Records"]
    x, y = st.columns(2)
    x.plotly_chart(px.bar(q, x="Network quality", y="Records", color="Network quality", title="Records per network-quality segment",
                          color_discrete_map=COLORS), use_container_width=True)
    hh = f.groupby("Hour")["NSI"].mean().reset_index()
    y.plotly_chart(px.line(hh, x="Hour", y="NSI", markers=True, title="Mean NSI by hour of day"), use_container_width=True)

# ---------------- Tab 2 ----------------
with t2:
    st.subheader("Efficiency distribution by network quality")
    ct = pd.crosstab(f["Network_Quality"], f["Efficiency_Status"], normalize="index").reindex(columns=EFF_ORDER).reset_index().melt("Network_Quality", value_name="Share")
    st.plotly_chart(px.bar(ct, x="Network_Quality", y="Share", color="Efficiency_Status", barmode="stack", color_discrete_map=COLORS,
                           category_orders={"Network_Quality": ["Low", "Medium", "High"]}, text_auto=".1%"), use_container_width=True)
    chi_ct = pd.crosstab(f["Network_Quality"], f["Efficiency_Status"])
    if chi_ct.shape[0] > 1 and chi_ct.shape[1] > 1:
        chi, p, _, _ = stats.chi2_contingency(chi_ct)
        v = np.sqrt(chi / (chi_ct.values.sum() * (min(chi_ct.shape) - 1)))
        st.info(f"Chi-square test: p = **{p:.3f}**, Cramér's V = **{v:.4f}** → " +
                ("no statistically meaningful association." if p > 0.05 or v < 0.05 else "statistically significant but practically small."))
    st.subheader("Latency vs efficiency")
    a, b = st.columns(2)
    lb = f.groupby("Latency_Band", observed=True).agg(Avg_Efficiency_Score=("Efficiency_Score", "mean"),
                                                       Speed=("Production_Speed_units_per_hr", "mean")).reset_index()
    a.plotly_chart(px.line(lb, x="Latency_Band", y="Avg_Efficiency_Score", markers=True, title="Efficiency score by latency band (0=Low, 2=High)"), use_container_width=True)
    b.plotly_chart(px.line(lb, x="Latency_Band", y="Speed", markers=True, title="Production speed by latency band (units/hr)"), use_container_width=True)
    samp = f.sample(min(6000, len(f)), random_state=1)
    sc_fig = px.scatter(samp, x="Network_Latency_ms", y="Production_Speed_units_per_hr", color="Efficiency_Status",
                        color_discrete_map=COLORS, opacity=.45, title="Latency vs production speed (6,000-point sample, OLS trend in black)")
    lr = stats.linregress(f["Network_Latency_ms"], f["Production_Speed_units_per_hr"])
    xs = np.array([f["Network_Latency_ms"].min(), f["Network_Latency_ms"].max()])
    sc_fig.add_scatter(x=xs, y=lr.intercept + lr.slope * xs, mode="lines", line=dict(color="black", width=3), name=f"OLS slope {lr.slope:+.3f} units/hr per ms")
    st.plotly_chart(sc_fig, use_container_width=True)
    st.subheader("Efficiency by operation mode & network quality (heatmap: % Medium+High)")
    hm = f.assign(good=f["Efficiency_Score"] >= 1).pivot_table(index="Operation_Mode", columns="Network_Quality", values="good", aggfunc="mean", observed=True)
    st.plotly_chart(px.imshow(hm, text_auto=".1%", color_continuous_scale="RdYlGn", aspect="auto"), use_container_width=True)
    st.subheader("Operation-mode interaction")
    mm = f.groupby("Operation_Mode").agg(Records=("NSI", "size"), Latency_vs_Speed_corr=("Network_Latency_ms", lambda s: s.corr(f.loc[s.index, "Production_Speed_units_per_hr"])),
                                         Latency_vs_Error_corr=("Network_Latency_ms", lambda s: s.corr(f.loc[s.index, "Error_Rate_%"])),
                                         Mean_Speed=("Production_Speed_units_per_hr", "mean"), Mean_Error=("Error_Rate_%", "mean")).round(4)
    st.dataframe(mm, use_container_width=True)
    st.subheader("Stable vs unstable network periods  |  real-time vs delayed communication")
    fs = mark_unstable(f)
    s1 = fs.groupby("Period_Stability")[["Production_Speed_units_per_hr", "Error_Rate_%", "Quality_Control_Defect_Rate_%", "Efficiency_Score"]].mean().round(3)
    s2 = f.groupby("Comm_Mode")[["Production_Speed_units_per_hr", "Error_Rate_%", "Quality_Control_Defect_Rate_%", "Efficiency_Score"]].mean().round(3)
    a, b = st.columns(2); a.dataframe(s1, use_container_width=True); b.dataframe(s2, use_container_width=True)

# ---------------- Tab 3 ----------------
with t3:
    st.subheader("Error rate vs packet loss")
    pb = f.groupby("PacketLoss_Band", observed=True).agg(Error=("Error_Rate_%", "mean"), Defect=("Quality_Control_Defect_Rate_%", "mean"),
                                                          n=("NSI", "size")).reset_index()
    a, b = st.columns(2)
    a.plotly_chart(px.bar(pb, x="PacketLoss_Band", y="Error", text_auto=".2f", title="Mean error rate (%) by packet-loss band", color_discrete_sequence=["#d62728"]), use_container_width=True)
    b.plotly_chart(px.bar(pb, x="PacketLoss_Band", y="Defect", text_auto=".2f", title="Mean defect rate (%) by packet-loss band", color_discrete_sequence=["#9467bd"]), use_container_width=True)
    r, rp = stats.spearmanr(f["Packet_Loss_%"], f["Error_Rate_%"])
    r2, rp2 = stats.spearmanr(f["Packet_Loss_%"], f["Quality_Control_Defect_Rate_%"])
    k = st.columns(2)
    k[0].metric("Spearman ρ: packet loss ↔ error rate", f"{r:+.4f}", f"p = {rp:.3f}", delta_color="off")
    k[1].metric("Spearman ρ: packet loss ↔ defect rate", f"{r2:+.4f}", f"p = {rp2:.3f}", delta_color="off")
    st.subheader("Defect rate during packet-loss spikes")
    thr = st.slider("Spike threshold (packet loss %)", 2.0, 4.9, 4.0, 0.1)
    sp = f.assign(Spike=np.where(f["Packet_Loss_%"] > thr, f"Spike (> {thr}%)", "Normal"))
    st.plotly_chart(px.box(sp, x="Spike", y="Quality_Control_Defect_Rate_%", color="Spike", points=False), use_container_width=True)
    g1, g2 = sp[sp["Spike"] == "Normal"], sp[sp["Spike"] != "Normal"]
    if len(g1) > 30 and len(g2) > 30:
        tt = stats.ttest_ind(g1["Quality_Control_Defect_Rate_%"], g2["Quality_Control_Defect_Rate_%"], equal_var=False)
        st.caption(f"Normal mean defect {g1['Quality_Control_Defect_Rate_%'].mean():.3f}% vs spike {g2['Quality_Control_Defect_Rate_%'].mean():.3f}% · Welch t-test p = {tt.pvalue:.3f}")
    st.subheader("Defect rate under combined network conditions")
    hm = f.pivot_table(index="PacketLoss_Band", columns="Latency_Band", values="Quality_Control_Defect_Rate_%", aggfunc="mean", observed=True)
    st.plotly_chart(px.imshow(hm, text_auto=".2f", color_continuous_scale="Reds", aspect="auto", labels=dict(x="Latency band (ms)", y="Packet-loss band (%)", color="Defect %")), use_container_width=True)

# ---------------- Tab 4 ----------------
with t4:
    st.subheader("Latency tolerance benchmarks")
    st.caption("Each band is compared with the best-performing (lowest-latency) band. A band is flagged only if the drop is statistically significant (Welch t-test, Bonferroni-adjusted) AND larger than the practical-significance margin.")
    margin = st.slider("Practical-significance margin (production speed, % of baseline)", 0.5, 10.0, 2.0, 0.5)

    def tolerance(col, bandcol, labels):
        rows = []
        avail = [l for l in labels if (f[bandcol] == l).sum() >= 50]
        if len(avail) < 2: return pd.DataFrame(columns=["Band", "Verdict", "Speed_drop_vs_best_pct"])
        labels = avail
        base = f[f[bandcol] == labels[0]]
        for lab in labels:
            g = f[f[bandcol] == lab]
            drop = (base["Production_Speed_units_per_hr"].mean() - g["Production_Speed_units_per_hr"].mean()) / base["Production_Speed_units_per_hr"].mean() * 100
            p = stats.ttest_ind(base["Production_Speed_units_per_hr"], g["Production_Speed_units_per_hr"], equal_var=False).pvalue if lab != labels[0] else 1.0
            good = (g["Efficiency_Score"] >= 1).mean()
            risk = "🔴 Risk" if (p < 0.05 / (len(labels) - 1) and drop > margin) else "🟢 Tolerable"
            rows.append(dict(Band=lab, Records=len(g), Avg_Speed=round(g["Production_Speed_units_per_hr"].mean(), 1), Speed_drop_vs_best_pct=round(drop, 2),
                             Medium_High_share=f"{good:.1%}", Avg_Error=round(g["Error_Rate_%"].mean(), 2), p_value=round(p, 3), Verdict=risk))
        return pd.DataFrame(rows)

    lt = tolerance("Network_Latency_ms", "Latency_Band", LAT_LABELS)
    st.dataframe(lt, use_container_width=True, hide_index=True)
    ok = lt[lt["Verdict"].astype(str).str.contains("Tolerable")]
    st.success(f"Highest latency band tested that stays within tolerance: **{ok['Band'].iloc[-1]} ms**" if len(ok) else "No tolerable band found.")
    st.subheader("Packet-loss risk zones")
    pl = tolerance("Packet_Loss_%", "PacketLoss_Band", PL_LABELS)
    st.dataframe(pl, use_container_width=True, hide_index=True)
    if len(pl) < 2: st.info("Not enough data in the current filter to build risk zones - widen filters.")
    else: st.plotly_chart(px.bar(pl, x="Band", y="Speed_drop_vs_best_pct", color="Verdict", title="Production-speed drop vs cleanest packet-loss band (%)",
                           color_discrete_map={"🟢 Tolerable": "#2E9E6B", "🔴 Risk": "#E4572E"}), use_container_width=True)
    st.subheader("What actually drives efficiency in this data?")
    @st.cache_data
    def drivers(n):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        cols = ["Network_Latency_ms", "Packet_Loss_%", "Temperature_C", "Vibration_Hz", "Power_Consumption_kW", "Predictive_Maintenance_Score",
                "Quality_Control_Defect_Rate_%", "Production_Speed_units_per_hr", "Error_Rate_%"]
        s = df.sample(20000, random_state=0)
        Xtr, Xte, ytr, yte = train_test_split(s[cols], s["Efficiency_Status"], test_size=.3, random_state=0)
        m = RandomForestClassifier(120, min_samples_leaf=20, n_jobs=-1, random_state=0, class_weight="balanced").fit(Xtr, ytr)
        pi = permutation_importance(m, Xte, yte, n_repeats=4, random_state=0, scoring="balanced_accuracy", n_jobs=-1)
        return pd.Series(pi.importances_mean, cols).sort_values()
    imp = drivers(len(df))
    st.plotly_chart(px.bar(imp, orientation="h", title="Permutation importance (drop in balanced accuracy)", labels=dict(value="Importance", index="Feature")), use_container_width=True)
    st.warning("**Honest finding:** across this dataset, latency and packet loss show no measurable effect on efficiency, speed, error or defect rate. "
               "Efficiency is determined almost entirely by Error_Rate and Production_Speed. Use the filters to confirm that this holds in every sub-segment.")
    st.markdown("**Recommendations**\n"
                "- Do not use latency/packet loss as a root-cause assumption for efficiency drops — investigate error rate and throughput first.\n"
                "- Keep SLA targets at ≤ 50 ms latency / ≤ 5 % loss as a *safe envelope* (no degradation observed inside it); do not over-invest to reduce latency further.\n"
                "- Re-test with real, time-correlated telemetry: add network-slice ID, jitter, retransmissions and per-machine timestamps.")

# ---------------- Tab 5 ----------------
with t5:
    st.markdown("""
**Dataset:** 100,000 machine-minute records · 50 machines · 1 Jan – 10 Mar 2025 · no missing values or duplicates.

**Network Stability Index (NSI)** = 100 × (1 − 0.5·Latency/50 − 0.5·PacketLoss/5). Network quality Low / Medium / High = tertiles of NSI.

**Latency Sensitivity Score** = OLS slope of efficiency score (Low 0, Medium 1, High 2) per ms (shown per +10 ms).
**Packet-Loss Impact Ratio** = (speed @ loss ≤ 1 % − speed @ loss > 4 %) / speed @ loss ≤ 1 %.
**Network-Efficiency Breakpoint** = latency threshold that maximises the fall in share of Medium+High efficiency.
**Unstable period** = hour whose mean NSI is in the bottom 25 % of hours. **Real-time** = latency ≤ 10 ms.
""")
    st.dataframe(f.drop(columns=["Date"]).head(200), use_container_width=True)
    st.download_button("⬇️ Download filtered data (CSV)", f.to_csv(index=False).encode(), "filtered_data.csv", "text/csv")
