import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import *
d = mark_unstable(load_data("Thales_Group_Manufacturing.csv")); R = json.load(open("results.json"))
C = {"Low":"#E4572E","Medium":"#F3A712","High":"#2E9E6B"}; plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
def save(n): plt.tight_layout(); plt.savefig(f"fig_{n}.png",dpi=160); plt.close()
fig,ax=plt.subplots(1,2,figsize=(9,3.2)); ax[0].hist(d.Network_Latency_ms,50,color="#1f77b4"); ax[0].set(title="Latency distribution",xlabel="ms")
ax[1].hist(d["Packet_Loss_%"],50,color="#d62728"); ax[1].set(title="Packet-loss distribution",xlabel="%"); save(1)
ct=pd.crosstab(d.Network_Quality,d.Efficiency_Status,normalize="index")[EFF_ORDER]
ax=ct.plot(kind="bar",stacked=True,color=[C[c] for c in EFF_ORDER],figsize=(6,3.6),rot=0); ax.set(title="Efficiency mix by network quality",xlabel="Network quality (NSI tertile)",ylabel="Share"); ax.legend(title="Efficiency",bbox_to_anchor=(1,1)); save(2)
fig,ax=plt.subplots(1,2,figsize=(9,3.4)); sp=d.groupby("Latency_Band",observed=True).Production_Speed_units_per_hr.mean()
ax[0].bar(sp.index.astype(str),sp.values,color="#1f77b4"); ax[0].set(title="Production speed by latency band",xlabel="Latency (ms)",ylabel="units/hr",ylim=(0,330))
ax[1].scatter(d.Network_Latency_ms.sample(4000,random_state=1),d.Production_Speed_units_per_hr.sample(4000,random_state=1),s=3,alpha=.4,color="grey"); ax[1].set(title="Latency vs speed (4k sample)",xlabel="Latency (ms)",ylabel="units/hr"); save(3)
g=d.groupby("PacketLoss_Band",observed=True)[["Error_Rate_%","Quality_Control_Defect_Rate_%"]].mean()
ax=g.plot(kind="bar",figsize=(6.5,3.4),rot=0,color=["#d62728","#9467bd"]); ax.set(title="Error & defect rate by packet-loss band",xlabel="Packet loss (%)",ylabel="mean %",ylim=(0,9)); ax.legend(["Error rate","Defect rate"]); save(4)
s=d.sample(6000,random_state=2); fig,ax=plt.subplots(figsize=(6,3.8))
for k in EFF_ORDER: q=s[s.Efficiency_Status==k]; ax.scatter(q["Error_Rate_%"],q.Production_Speed_units_per_hr,s=4,alpha=.5,color=C[k],label=k)
ax.set(title="How efficiency is actually assigned",xlabel="Error rate (%)",ylabel="Production speed (units/hr)"); ax.legend(markerscale=4); save(5)
pi=pd.Series(R["perm_imp"]).sort_values(); fig,ax=plt.subplots(figsize=(6,3.6)); ax.barh(pi.index,pi.values,color=["#1f77b4" if i in("Network_Latency_ms","Packet_Loss_%") else "grey" for i in pi.index]); ax.set(title="Permutation importance for predicting efficiency",xlabel="Drop in balanced accuracy"); save(6)
