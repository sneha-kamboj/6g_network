import json, numpy as np, pandas as pd
from scipy import stats
from utils import *
d = mark_unstable(load_data("/mnt/user-data/uploads/Thales_Group_Manufacturing.csv"))
R = {}
R["n"] = len(d); R["span"] = [str(d.Datetime.min()), str(d.Datetime.max())]
R["eff_share"] = d.Efficiency_Status.value_counts(normalize=True).round(4).to_dict()
R["lat_sens"] = latency_sensitivity(d); R["pl_ratio"] = packet_loss_impact_ratio(d)
R["breakpoint_lat"] = efficiency_breakpoint(d); R["breakpoint_pl"] = efficiency_breakpoint(d, "Packet_Loss_%", .1)
# Spearman
num = ["Network_Latency_ms","Packet_Loss_%","NSI"]
tg = ["Efficiency_Score","Production_Speed_units_per_hr","Error_Rate_%","Quality_Control_Defect_Rate_%"]
R["spearman"] = {a:{b:[round(stats.spearmanr(d[a],d[b])[0],4), float(stats.spearmanr(d[a],d[b])[1])] for b in tg} for a in num}
# chi2
for k,c in [("chi_nq","Network_Quality"),("chi_lat","Latency_Band"),("chi_pl","PacketLoss_Band"),("chi_mode","Operation_Mode"),("chi_stab","Period_Stability")]:
    ct = pd.crosstab(d[c], d.Efficiency_Status); chi,p,dof,_ = stats.chi2_contingency(ct)
    R[k] = dict(chi2=chi,p=p,dof=int(dof),cramers_v=float(np.sqrt(chi/(ct.values.sum()*(min(ct.shape)-1)))))
# Kruskal speed/error by latency band
R["kw_speed_lat"] = stats.kruskal(*[g.Production_Speed_units_per_hr for _,g in d.groupby("Latency_Band",observed=True)])._asdict() if hasattr(stats.kruskal(d.Network_Latency_ms,d.NSI),'_asdict') else None
R["kw_err_pl"] = list(stats.kruskal(*[g["Error_Rate_%"] for _,g in d.groupby("PacketLoss_Band",observed=True)]))
R["kw_def_pl"] = list(stats.kruskal(*[g["Quality_Control_Defect_Rate_%"] for _,g in d.groupby("PacketLoss_Band",observed=True)]))
R["kw_speed_lat"] = list(stats.kruskal(*[g.Production_Speed_units_per_hr for _,g in d.groupby("Latency_Band",observed=True)]))
# tables
R["eff_by_nq"] = pd.crosstab(d.Network_Quality,d.Efficiency_Status,normalize="index").round(4).to_dict("index")
R["speed_by_lat"] = d.groupby("Latency_Band",observed=True).Production_Speed_units_per_hr.mean().round(2).to_dict()
R["err_by_pl"] = d.groupby("PacketLoss_Band",observed=True)[["Error_Rate_%","Quality_Control_Defect_Rate_%"]].mean().round(3).to_dict("index")
R["eff_by_mode"] = pd.crosstab(d.Operation_Mode,d.Efficiency_Status,normalize="index").round(4).to_dict("index")
R["eff_by_stab"] = pd.crosstab(d.Period_Stability,d.Efficiency_Status,normalize="index").round(4).to_dict("index")
R["comm"] = d.groupby("Comm_Mode")[["Production_Speed_units_per_hr","Error_Rate_%"]].mean().round(3).to_dict("index")
R["net_means"] = d[["Network_Latency_ms","Packet_Loss_%","NSI"]].describe().round(3).to_dict()
# ML: what predicts efficiency?
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.inspection import permutation_importance
mech = ["Temperature_C","Vibration_Hz","Power_Consumption_kW","Predictive_Maintenance_Score"]
net = ["Network_Latency_ms","Packet_Loss_%"]
out = ["Quality_Control_Defect_Rate_%","Production_Speed_units_per_hr","Error_Rate_%"]
s = d.sample(30000, random_state=1); y = s.Efficiency_Status
def cv(cols): return float(cross_val_score(RandomForestClassifier(150,min_samples_leaf=20,n_jobs=-1,random_state=0,class_weight="balanced"),s[cols],y,cv=3,scoring="balanced_accuracy").mean())
R["bal_acc"] = {"network_only":cv(net),"mechanical_only":cv(mech),"network+mechanical":cv(net+mech),"outputs(speed,error,defect)":cv(out),"all":cv(net+mech+out)}
Xtr,Xte,ytr,yte = train_test_split(s[net+mech+out],y,test_size=.3,random_state=0)
rf = RandomForestClassifier(150,min_samples_leaf=20,n_jobs=-1,random_state=0,class_weight="balanced").fit(Xtr,ytr)
pi = permutation_importance(rf,Xte,yte,n_repeats=5,random_state=0,scoring="balanced_accuracy",n_jobs=-1)
R["perm_imp"] = dict(sorted(zip(net+mech+out,pi.importances_mean.round(4)),key=lambda x:-x[1]))
# how efficiency is defined (rule discovery)
g=d.groupby("Efficiency_Status")[["Error_Rate_%","Production_Speed_units_per_hr"]].agg(["min","max"]).round(2)
R["eff_rule"]={k:{"_".join(c):float(v) for c,v in r.items()} for k,r in g.iterrows()}
json.dump(R,open("results.json","w"),indent=1,default=float)
print(json.dumps(R,indent=1,default=float))
