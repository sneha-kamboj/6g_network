"""Shared data + KPI logic for the 6G Smart Factory project."""
import numpy as np, pandas as pd

EFF_ORDER = ["Low", "Medium", "High"]
EFF_SCORE = {"Low": 0, "Medium": 1, "High": 2}
LAT_BINS = [0, 10, 20, 30, 40, 50]
LAT_LABELS = ["0-10", "10-20", "20-30", "30-40", "40-50"]
PL_BINS = [-0.001, 1, 2, 3, 4, 5]
PL_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-5"]
LAT_MAX, PL_MAX = 50.0, 5.0


def load_data(path):
    d = pd.read_csv(path)
    d["Datetime"] = pd.to_datetime(d["Date"] + " " + d["Timestamp"], format="%d-%m-%Y %H:%M:%S")
    d = d.drop(columns=["Date", "Timestamp"]).sort_values("Datetime").reset_index(drop=True)
    d["Date"] = d["Datetime"].dt.date
    d["Hour"] = d["Datetime"].dt.hour
    d["Efficiency_Score"] = d["Efficiency_Status"].map(EFF_SCORE)
    # Network Stability Index: 100 = perfect network, 0 = worst observed latency AND packet loss
    d["NSI"] = 100 * (1 - 0.5 * d["Network_Latency_ms"] / LAT_MAX - 0.5 * d["Packet_Loss_%"] / PL_MAX)
    q1, q2 = d["NSI"].quantile([1 / 3, 2 / 3])
    d["Network_Quality"] = pd.cut(d["NSI"], [-np.inf, q1, q2, np.inf], labels=["Low", "Medium", "High"])
    d["Latency_Band"] = pd.cut(d["Network_Latency_ms"], LAT_BINS, labels=LAT_LABELS, include_lowest=True)
    d["PacketLoss_Band"] = pd.cut(d["Packet_Loss_%"], PL_BINS, labels=PL_LABELS)
    d["Comm_Mode"] = np.where(d["Network_Latency_ms"] <= 10, "Real-time (<=10 ms)", "Delayed (>10 ms)")
    return d


def mark_unstable(d, pct=25):
    """Hourly window is 'unstable' if its mean NSI falls in bottom `pct` percent of hourly windows."""
    d = d.copy()
    key = d["Datetime"].dt.floor("h")
    h = d.groupby(key)["NSI"].mean()
    thr = np.percentile(h, pct)
    d["Period_Stability"] = np.where(key.map(h) <= thr, "Unstable", "Stable")
    return d


def latency_sensitivity(d):
    """Slope of efficiency score (0-2) per +1 ms latency, and units/hr per +1 ms."""
    from scipy import stats
    r1 = stats.linregress(d["Network_Latency_ms"], d["Efficiency_Score"])
    r2 = stats.linregress(d["Network_Latency_ms"], d["Production_Speed_units_per_hr"])
    return dict(eff_slope=r1.slope, eff_p=r1.pvalue, speed_slope=r2.slope, speed_p=r2.pvalue,
                eff_per_10ms=r1.slope * 10, speed_per_10ms=r2.slope * 10)


def packet_loss_impact_ratio(d):
    lo = d[d["Packet_Loss_%"] <= 1]; hi = d[d["Packet_Loss_%"] > 4]
    if len(lo) == 0 or len(hi) == 0:
        return np.nan
    return (lo["Production_Speed_units_per_hr"].mean() - hi["Production_Speed_units_per_hr"].mean()) / lo["Production_Speed_units_per_hr"].mean()


def efficiency_breakpoint(d, col="Network_Latency_ms", step=1.0):
    """Scan thresholds; return threshold with largest drop in share of High+Medium efficiency (above vs below)."""
    good = (d["Efficiency_Score"] >= 1).astype(float)
    best = (np.nan, 0.0)
    for t in np.arange(d[col].quantile(.05), d[col].quantile(.95), step):
        a, b = good[d[col] <= t], good[d[col] > t]
        if len(a) > 200 and len(b) > 200:
            drop = a.mean() - b.mean()
            if drop > best[1]:
                best = (t, drop)
    return best
