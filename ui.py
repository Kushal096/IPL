from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ipl_model.pkl"
DELIVERIES_PATH = BASE_DIR / "CSV" / "deliveries.csv"
MATCHES_PATH = BASE_DIR / "CSV" / "matches.csv"


TEAMS = [
	"Sunrisers Hyderabad",
	"Mumbai Indians",
	"Royal Challengers Bengaluru",
	"Kolkata Knight Riders",
	"Punjab Kings",
	"Chennai Super Kings",
	"Rajasthan Royals",
	"Delhi Capitals",
	"Gujarat Titans",
	"Lucknow Super Giants",
]


NAME_MAP = {
	"Delhi Daredevils": "Delhi Capitals",
	"Deccan Chargers": "Sunrisers Hyderabad",
	"Kings XI Punjab": "Punjab Kings",
	"Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}


VENUE_MAPPING = {
	"feroz shah kotla": "arun jaitley stadium",
	"arun jaitley stadium": "arun jaitley stadium",
	"sardar patel stadium": "narendra modi stadium",
	"motera": "narendra modi stadium",
	"narendra modi stadium": "narendra modi stadium",
	"punjab cricket association stadium": "punjab cricket association is bindra stadium",
	"punjab cricket association is bindra stadium": "punjab cricket association is bindra stadium",
	"subrata roy sahara stadium": "maharashtra cricket association stadium",
	"maharashtra cricket association stadium": "maharashtra cricket association stadium",
	"sheikh zayed stadium": "zayed cricket stadium",
	"zayed cricket stadium": "zayed cricket stadium",
	"ma chidambaram stadium": "ma chidambaram stadium",
	"wankhede stadium": "wankhede stadium",
	"brabourne stadium": "brabourne stadium",
	"dr dy patil sports academy": "dr dy patil sports academy",
	"rajiv gandhi international stadium": "rajiv gandhi international stadium",
	"dr. y.s. rajasekhara reddy aca-vdca cricket stadium": "dr. y.s. rajasekhara reddy aca-vdca cricket stadium",
	"sawai mansingh stadium": "sawai mansingh stadium",
	"vidarbha cricket association stadium": "vidarbha cricket association stadium",
	"bharat ratna shri atal bihari vajpayee ekana cricket stadium": "bharat ratna shri atal bihari vajpayee ekana cricket stadium",
	"jsca international stadium complex": "jsca international stadium complex",
	"shaheed veer narayan singh international stadium": "shaheed veer narayan singh international stadium",
	"holkar cricket stadium": "holkar cricket stadium",
	"m chinnaswamy stadium": "m chinnaswamy stadium",
	"eden gardens": "eden gardens",
	"newlands": "newlands",
	"kingsmead": "kingsmead",
	"supersport park": "supersport park",
	"st george's park": "st george's park",
	"new wanderers stadium": "new wanderers stadium",
	"buffalo park": "buffalo park",
	"outsurance oval": "outsurance oval",
	"de beers diamond oval": "de beers diamond oval",
	"barsapara cricket stadium": "barsapara cricket stadium",
	"sharjah cricket stadium": "sharjah cricket stadium",
	"dubai international cricket stadium": "dubai international cricket stadium",
	"maharaja yadavindra singh international cricket stadium": "maharaja yadavindra singh international cricket stadium",
}


@st.cache_resource
def load_model():
	with open(MODEL_PATH, "rb") as model_file:
		return pickle.load(model_file)


@st.cache_data(show_spinner=False)
def load_reference_data():
	deliveries = pd.read_csv(
		DELIVERIES_PATH,
		usecols=["match_id", "batting_team", "batter", "batsman_runs", "ball", "extras_type"],
	)
	matches = pd.read_csv(MATCHES_PATH, usecols=["venue"])

	deliveries["batting_team"] = deliveries["batting_team"].replace(NAME_MAP)
	deliveries = deliveries[deliveries["batting_team"].isin(TEAMS)].copy()

	matches["venue"] = (
		matches["venue"]
		.astype(str)
		.str.strip()
		.str.lower()
		.str.split(",")
		.str[0]
		.replace(VENUE_MAPPING)
	)
	venue_options = sorted(matches["venue"].dropna().unique().tolist())

	valid_balls = deliveries[
		deliveries["extras_type"].isna() | deliveries["extras_type"].isin(["legbyes", "byes", "noballs"])
	].copy()

	per_match = (
		valid_balls.groupby(["batter", "match_id"]).agg(
			runs_in_match=("batsman_runs", "sum"),
			balls_in_match=("ball", "count"),
		)
	).reset_index()

	per_match = per_match.sort_values(["batter", "match_id"]).copy()
	per_match["overall_runs"] = per_match.groupby("batter")["runs_in_match"].transform(
		lambda series: series.cumsum().shift(fill_value=0)
	)
	per_match["overall_balls"] = per_match.groupby("batter")["balls_in_match"].transform(
		lambda series: series.cumsum().shift(fill_value=0)
	)

	per_match["overall_SR"] = ((per_match["overall_runs"] / per_match["overall_balls"]) * 100).round(2)

	per_match["last5_runs"] = per_match.groupby("batter")["runs_in_match"].transform(
		lambda series: series.shift(fill_value=0).rolling(5, min_periods=1).sum()
	)
	per_match["last5_balls"] = per_match.groupby("batter")["balls_in_match"].transform(
		lambda series: series.shift(fill_value=0).rolling(5, min_periods=1).sum()
	)
	per_match["last5_SR"] = ((per_match["last5_runs"] / per_match["last5_balls"]) * 100).round(2)

	per_match[["overall_SR", "last5_SR"]] = per_match[["overall_SR", "last5_SR"]].fillna(0)
	per_match = per_match.sort_values(["batter", "match_id"]).groupby("batter", as_index=False).tail(1)

	team_batters = (
		deliveries[["batting_team", "batter"]]
		.dropna()
		.drop_duplicates()
		.groupby("batting_team")["batter"]
		.apply(lambda names: sorted(names.unique().tolist()))
		.to_dict()
	)

	return venue_options, per_match, team_batters


def phase_from_balls_left(balls_left: int) -> str:
	if balls_left <= 30:
		return "death_over"
	if balls_left >= 84:
		return "powerplay"
	return "middle_over"


def build_match_features(
	batting_team: str,
	bowling_team: str,
	balls_bowled: int,
	current_runs: int,
	target_runs: int,
	wickets_left: int,
	striker: str,
	venue: str,
	player_data: pd.DataFrame,
) -> pd.DataFrame:
	balls_left = max(1, 120 - balls_bowled)
	runs_left = max(0, target_runs - current_runs)
	overs_completed = balls_bowled / 6

	crr = (current_runs * 6) / max(1, balls_bowled)
	rrr = (runs_left * 6) / balls_left

	striker_rows = player_data[player_data["batter"] == striker]
	if striker_rows.empty:
		overall_runs = 0
		overall_balls = 0
		overall_sr = 0
		last5_runs = 0
		last5_balls = 0
		last5_sr = 0
	else:
		striker_stats = striker_rows.iloc[-1]
		overall_runs = float(striker_stats["overall_runs"])
		overall_balls = float(striker_stats["overall_balls"])
		overall_sr = float(striker_stats["overall_SR"])
		last5_runs = float(striker_stats["last5_runs"])
		last5_balls = float(striker_stats["last5_balls"])
		last5_sr = float(striker_stats["last5_SR"])

	match_data = {
		"batting_team": [batting_team],
		"bowling_team": [bowling_team],
		"venue": [venue],
		"runs_left": [runs_left],
		"balls_left": [balls_left],
		"wickets_left": [wickets_left],
		"target_runs": [target_runs],
		"crr": [crr],
		"rrr": [rrr],
		"overall_runs": [overall_runs],
		"overall_balls": [overall_balls],
		"overall_SR": [overall_sr],
		"last5_runs": [last5_runs],
		"last5_balls": [last5_balls],
		"last5_SR": [last5_sr],
		"current_phase": [phase_from_balls_left(balls_left)],
		"over_completed": [overs_completed],
	}
	return pd.DataFrame(match_data)


st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")

st.markdown("## IPL Chase Win Probability Predictor")
st.caption("Enter the current chase situation to estimate winning chances for both teams.")

if not MODEL_PATH.exists():
	st.error("Model file not found: ipl_model.pkl")
	st.stop()

if not DELIVERIES_PATH.exists() or not MATCHES_PATH.exists():
	st.error("CSV files not found. Please keep deliveries.csv and matches.csv inside CSV/ folder.")
	st.stop()

with st.spinner("Loading model and reference data..."):
	model = load_model()
	venues, player_stats, team_batters_map = load_reference_data()

st.sidebar.header("Input Guide")
st.sidebar.write("- Use completed balls in 2nd innings (0 to 119).")
st.sidebar.write("- Choose striker from the batting team player list.")
st.sidebar.write("- App auto-computes phase, rates, and batter form features.")

left_col, right_col = st.columns(2)

with left_col:
	batting_team = st.selectbox("Batting Team", options=TEAMS, index=2)
	bowling_options = [team for team in TEAMS if team != batting_team]
	bowling_team = st.selectbox("Bowling Team", options=bowling_options, index=0)

	venue = st.selectbox(
		"Venue",
		options=venues if venues else sorted(set(VENUE_MAPPING.values())),
	)

	batter_options = team_batters_map.get(batting_team, [])
	if not batter_options:
		batter_options = sorted(player_stats["batter"].dropna().unique().tolist())
	striker = st.selectbox("Striker", options=batter_options)

with right_col:
	balls_bowled = st.slider("Balls Bowled", min_value=0, max_value=119, value=96, step=1)
	current_runs = st.number_input("Current Runs", min_value=0, max_value=300, value=140, step=1)
	target_runs = st.number_input("Target Runs", min_value=1, max_value=300, value=180, step=1)
	wickets_left = st.slider("Wickets Left", min_value=0, max_value=10, value=5, step=1)

predict_clicked = st.button("Predict Win Probability", type="primary", use_container_width=True)

if predict_clicked:
	if target_runs <= current_runs:
		st.success("Batting team has already reached/passed the target. Win probability is 100%.")
		st.stop()

	feature_row = build_match_features(
		batting_team=batting_team,
		bowling_team=bowling_team,
		balls_bowled=balls_bowled,
		current_runs=current_runs,
		target_runs=target_runs,
		wickets_left=wickets_left,
		striker=striker,
		venue=venue,
		player_data=player_stats,
	)

	probabilities = model.predict_proba(feature_row)[0]
	bowling_win_pct = float(probabilities[0]) * 100
	batting_win_pct = float(probabilities[1]) * 100

	st.markdown("### Predicted Winning Chances")
	metric_col1, metric_col2 = st.columns(2)
	metric_col1.metric(f"{batting_team}", f"{batting_win_pct:.2f}%")
	metric_col2.metric(f"{bowling_team}", f"{bowling_win_pct:.2f}%")

	st.progress(min(max(int(round(batting_win_pct)), 0), 100), text=f"{batting_team} win chance")
	st.progress(min(max(int(round(bowling_win_pct)), 0), 100), text=f"{bowling_team} win chance")

	with st.expander("Computed Match Snapshot"):
		st.dataframe(feature_row, use_container_width=True)
