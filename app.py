"""Launch from repository root: streamlit run app.py"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from nba_predictor.data import utc
from nba_predictor.predict import forecast, upcoming_targets

st.set_page_config(page_title="NBA Next Game Predictor", page_icon="🏀", layout="wide")
st.title("🏀 NBA Next Game Predictor")
mode = st.sidebar.selectbox("Dataset", ["Synthetic demonstration", "Real NBA data"])
synthetic = mode.startswith("Synthetic")
root = Path("data/synthetic" if synthetic else "data")
artifacts = Path("artifacts/synthetic" if synthetic else "artifacts")
reports = Path("reports/synthetic" if synthetic else "reports/real")
if synthetic:
    st.warning("Artificial players and games. These results do not measure NBA prediction performance.")
if not (root/"clean"/"logs.parquet").exists() or not (artifacts/"model.joblib").exists():
    st.info("Run the offline demo or the ingestion and training commands in the README first.")
    st.stop()
logs = pd.read_parquet(root/"clean"/"logs.parquet")
model = joblib.load(artifacts/"model.joblib")
players = logs[["player_id", "player_name"]].drop_duplicates("player_id")
labels = dict(zip(players.player_id.astype(str), players.player_name))
player = st.selectbox("Player", labels, format_func=lambda x: labels[x])
threshold = st.number_input("Points threshold (probability of scoring more than this)", value=20.5, step=.5)
selected = logs[logs.player_id.astype(str) == player].sort_values("tipoff")
st.caption(f"Statistics retrieved: {selected.retrieved_at.max()} · model executed: {model.trained_at}")
if (root/"clean"/"schedule.parquet").exists():
    schedule = pd.read_parquet(root/"clean"/"schedule.parquet")
    as_of = utc(schedule.tipoff.max())-pd.Timedelta(hours=1) if synthetic else None
    targets = upcoming_targets(logs, schedule, player, as_of)
    if not targets.empty:
        chosen = st.selectbox("Upcoming game", range(len(targets)),
                              format_func=lambda i: f"{targets.iloc[i].game_id} · {targets.iloc[i].tipoff}")
        if st.button("Forecast points"):
            try:
                result = forecast(model, logs, targets.iloc[chosen].to_dict(), root, root/"text_index.parquet", threshold, as_of)
                f = result["forecast"]
                a, b, c = st.columns(3)
                a.metric("Expected points", round(f["expected_points"], 1))
                b.metric("Median points", round(f["median_points"], 1))
                c.metric("Estimated minutes if playing", round(f["estimated_minutes"], 1))
                st.write("Prediction intervals", f["prediction_intervals"])
                st.write(f"Probability of more than {threshold}: {f['probability_exceeding_threshold']:.1%}")
                st.caption("Conditional on playing. Availability probability is unknown. Coverage is a target, not a guarantee.")
                with st.expander("Sources, assumptions, missing information and timestamps", expanded=True):
                    st.json(result)
            except ValueError as exc:
                st.info(str(exc))
    else:
        st.info("No upcoming game is available in this schedule snapshot.")
else:
    st.info("Upcoming schedule is missing. Fetch or import a timestamped schedule as described in the README.")
st.subheader("Historical played-game points")
st.line_chart(selected[selected.played].set_index("tipoff")[["points"]])
st.subheader("Measured evaluation")
if (reports/"evaluation.json").exists():
    report = json.loads((reports/"evaluation.json").read_text())
    st.write(report["data_kind"])
    st.json({"metrics": report["metrics"], "goals": report["goals"], "counts": report["counts"]})
    if (reports/"reliability.png").exists():
        st.image(str(reports/"reliability.png"))
