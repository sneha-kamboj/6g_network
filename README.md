# 6G Network Performance vs Manufacturing Efficiency (Thales Group / Unified Mentor)

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Keep `Thales_Group_Manufacturing.csv` in the same folder (or upload it from the sidebar).

## Files
- `app.py` — Streamlit dashboard (4 modules + Method tab, filters: network quality, efficiency class, operation mode, time window)
- `utils.py` — NSI, network bands, KPI functions
- `analyze.py` / `figs.py` — statistical analysis and figures used in the paper (`results.json`)
- `Research_Paper_6G_Manufacturing_Efficiency.docx`, `Executive_Summary_Government_Stakeholders.docx`
