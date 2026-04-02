from pathlib import Path
import pickle
import joblib

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ipl_model.joblib"
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


BOWLER_WICKET_KINDS = {
	"bowled",
	"caught",
	"caught and bowled",
	"lbw",
	"stumped",
	"hit wicket",
}


@st.cache_resource
def load_model():
	# with open(MODEL_PATH, "rb") as model_file:
	# 	return pickle.load(model_file)
	with open(MODEL_PATH, "rb") as model_file:
		return joblib.load(model_file)


def predict_with_model(model, feature_row: pd.DataFrame):
	if hasattr(model, "predict_proba"):
		return model.predict_proba(feature_row)[0]

	if isinstance(model, dict):
		# New package format: explicit fixed shares
		other_model = model.get("other_model")
		rrr_model = model.get("rrr_model")
		wickets_model = model.get("wickets_model")
		if other_model is not None and rrr_model is not None and wickets_model is not None:
			other_features = model.get("other_features")
			rrr_feature = model.get("rrr_feature", ["rrr"])
			wickets_feature = model.get("wickets_feature", ["wickets_left"])

			if other_features is None:
				other_features = [col for col in feature_row.columns if col not in ["rrr", "wickets_left"]]

			other_weight = float(model.get("other_weight", 0.50))
			rrr_weight = float(model.get("rrr_weight", 0.25))
			wickets_weight = float(model.get("wickets_weight", 0.25))

			total_weight = other_weight + rrr_weight + wickets_weight
			if total_weight <= 0:
				other_weight, rrr_weight, wickets_weight = 0.50, 0.25, 0.25
			else:
				other_weight = other_weight / total_weight
				rrr_weight = rrr_weight / total_weight
				wickets_weight = wickets_weight / total_weight

			other_prob = float(other_model.predict_proba(feature_row[other_features])[0][1])
			rrr_prob = float(rrr_model.predict_proba(feature_row[rrr_feature])[0][1])
			wickets_prob = float(wickets_model.predict_proba(feature_row[wickets_feature])[0][1])

			batting_prob = (
				(other_weight * other_prob)
				+ (rrr_weight * rrr_prob)
				+ (wickets_weight * wickets_prob)
			)
			bowling_prob = 1 - batting_prob
			return [bowling_prob, batting_prob]

		# Backward-compatible package format: 50% main + 50% key features model
		main_model = model.get("main_model")
		key_model = model.get("key_model")
		key_features = model.get("key_features", ["rrr", "wickets_left"])
		main_weight = float(model.get("main_weight", 0.5))
		key_weight = float(model.get("key_weight", 0.5))

		if main_model is None or key_model is None:
			raise ValueError("Invalid model package: missing main_model or key_model")

		total_weight = main_weight + key_weight
		if total_weight <= 0:
			main_weight, key_weight = 0.5, 0.5
		else:
			main_weight = main_weight / total_weight
			key_weight = key_weight / total_weight

		main_prob = float(main_model.predict_proba(feature_row)[0][1])
		key_prob = float(key_model.predict_proba(feature_row[key_features])[0][1])
		batting_prob = (main_weight * main_prob) + (key_weight * key_prob)
		bowling_prob = 1 - batting_prob
		return [bowling_prob, batting_prob]

	raise ValueError("Unsupported model format in ipl_model.joblib")


@st.cache_data(show_spinner=False)
def load_reference_data():
	deliveries = pd.read_csv(DELIVERIES_PATH)
	matches = pd.read_csv(MATCHES_PATH, usecols=["id", "venue", "date", "method", "team1", "team2"])
	matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

	deliveries["batting_team"] = deliveries["batting_team"].replace(NAME_MAP)
	deliveries["bowling_team"] = deliveries["bowling_team"].replace(NAME_MAP)
	deliveries = deliveries[
		deliveries["batting_team"].isin(TEAMS) & deliveries["bowling_team"].isin(TEAMS)
	].copy()
	deliveries = deliveries[deliveries["inning"].isin([1, 2])].copy()

	matches["team1"] = matches["team1"].replace(NAME_MAP)
	matches["team2"] = matches["team2"].replace(NAME_MAP)
	valid_match_ids = matches[
		matches["method"].isna() & matches["team1"].isin(TEAMS) & matches["team2"].isin(TEAMS)
	]["id"]

	model_deliveries = deliveries[
		(deliveries["inning"] == 2) & deliveries["match_id"].isin(valid_match_ids)
	].copy()

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
	match_info = matches[["id", "date"]].rename(columns={"id": "match_id"}).copy()

	model_deliveries["is_legal_ball"] = model_deliveries["extras_type"] != "wides"
	model_deliveries["bowler_legal_ball"] = ~model_deliveries["extras_type"].isin(["wides", "noballs"])
	model_deliveries["bowler_wicket"] = (
		(model_deliveries["is_wicket"] == 1)
		& model_deliveries["dismissal_kind"].isin(BOWLER_WICKET_KINDS)
	).astype(int)
	model_deliveries["bowler_runs_conceded"] = (
		model_deliveries["batsman_runs"]
		+ model_deliveries["extra_runs"].where(~model_deliveries["extras_type"].isin(["byes", "legbyes"]), 0)
	)

	valid_balls = model_deliveries[model_deliveries["is_legal_ball"]].copy()

	per_match = (
		valid_balls.groupby(["batter", "match_id"]).agg(
			runs_in_match=("batsman_runs", "sum"),
			balls_in_match=("is_legal_ball", "sum"),
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

	# Calculate bowler statistics per match
	per_match_bowler = (
		model_deliveries.groupby(["bowler", "match_id"]).agg(
			wickets_in_match=("bowler_wicket", "sum"),
			runs_conceded_in_match=("bowler_runs_conceded", "sum"),
			balls_bowled_in_match=("bowler_legal_ball", "sum"),
		)
	).reset_index()
	per_match_bowler = per_match_bowler.sort_values(["bowler", "match_id"]).copy()

	# Calculate cumulative bowler statistics
	per_match_bowler["overall_wickets"] = (
		per_match_bowler.groupby("bowler")["wickets_in_match"]
		.transform(lambda x: x.cumsum().shift(fill_value=0))
	)
	per_match_bowler["overall_runs_conceded"] = (
		per_match_bowler.groupby("bowler")["runs_conceded_in_match"]
		.transform(lambda x: x.cumsum().shift(fill_value=0))
	)
	per_match_bowler["overall_balls_bowled"] = (
		per_match_bowler.groupby("bowler")["balls_bowled_in_match"]
		.transform(lambda x: x.cumsum().shift(fill_value=0))
	)

	# Calculate bowling average (runs conceded per wicket)
	per_match_bowler["bowling_avg"] = (
		(per_match_bowler["overall_runs_conceded"] / per_match_bowler["overall_wickets"])
		.replace([float('inf'), float('-inf')], 0)
		.fillna(0)
		.round(2)
	)

	# Calculate bowling economy (runs conceded per over)
	per_match_bowler["bowling_economy"] = (
		(per_match_bowler["overall_runs_conceded"] * 6 / per_match_bowler["overall_balls_bowled"])
		.replace([float('inf'), float('-inf')], 0)
		.fillna(0)
		.round(2)
	)

	# Calculate last 5 matches bowler statistics
	per_match_bowler["last5_wickets"] = (
		per_match_bowler.groupby("bowler")["wickets_in_match"]
		.transform(lambda x: x.shift(fill_value=0).rolling(5, min_periods=1).sum())
	)
	per_match_bowler["last5_runs"] = (
		per_match_bowler.groupby("bowler")["runs_conceded_in_match"]
		.transform(lambda x: x.shift(fill_value=0).rolling(5, min_periods=1).sum())
	)
	per_match_bowler["last5_balls"] = (
		per_match_bowler.groupby("bowler")["balls_bowled_in_match"]
		.transform(lambda x: x.shift(fill_value=0).rolling(5, min_periods=1).sum())
	)

	# Last 5 matches bowling economy
	per_match_bowler["last5_economy"] = (
		(per_match_bowler["last5_runs"] * 6 / per_match_bowler["last5_balls"])
		.replace([float('inf'), float('-inf')], 0)
		.fillna(0)
		.round(2)
	)

	team_batters = (
		deliveries[["batting_team", "batter"]]
		.dropna()
		.drop_duplicates()
		.groupby("batting_team")["batter"]
		.apply(lambda names: sorted(names.unique().tolist()))
		.to_dict()
	)

	return venue_options, per_match, team_batters, deliveries, per_match_bowler, match_info


@st.cache_data(show_spinner=False)
def get_batter_last5_games(batter_name: str, all_deliveries: pd.DataFrame, match_info: pd.DataFrame) -> dict:
	"""Get batter's last 5 games statistics"""
	batter_data = all_deliveries[all_deliveries["batter"] == batter_name].copy()
	batter_data["is_legal_ball"] = batter_data["extras_type"] != "wides"
	
	# Get matches where batter played
	batter_matches = (
		batter_data.groupby("match_id").agg(
			runs=("batsman_runs", "sum"),
			balls=("is_legal_ball", "sum"),
			fours=("batsman_runs", lambda x: (x == 4).sum()),
			sixes=("batsman_runs", lambda x: (x == 6).sum()),
		)
	).reset_index()
	
	batter_matches = batter_matches.merge(match_info, on="match_id", how="left")
	batter_matches = batter_matches.sort_values(["date", "match_id"]).tail(5).copy()
	batter_matches["SR"] = ((batter_matches["runs"] / batter_matches["balls"].replace(0, pd.NA)) * 100).fillna(0).round(2)
	batter_matches["date"] = batter_matches["date"].dt.strftime("%Y-%m-%d")
	
	if len(batter_matches) == 0:
		return {"games": [], "avg_runs": 0, "avg_sr": 0, "total_4s": 0, "total_6s": 0}
	
	return {
		"games": batter_matches.to_dict("records"),
		"avg_runs": batter_matches["runs"].mean(),
		"avg_sr": batter_matches["SR"].mean(),
		"total_4s": batter_matches["fours"].sum(),
		"total_6s": batter_matches["sixes"].sum(),
	}


@st.cache_data(show_spinner=False)
def get_bowler_last5_games(bowler_name: str, all_deliveries: pd.DataFrame, match_info: pd.DataFrame) -> dict:
	"""Get bowler's last 5 games statistics"""
	bowler_data = all_deliveries[all_deliveries["bowler"] == bowler_name].copy()
	bowler_data["bowler_legal_ball"] = ~bowler_data["extras_type"].isin(["wides", "noballs"])
	bowler_data["bowler_wicket"] = (
		(bowler_data["is_wicket"] == 1)
		& bowler_data["dismissal_kind"].isin(BOWLER_WICKET_KINDS)
	).astype(int)
	bowler_data["bowler_runs_conceded"] = (
		bowler_data["batsman_runs"]
		+ bowler_data["extra_runs"].where(~bowler_data["extras_type"].isin(["byes", "legbyes"]), 0)
	)
	
	# Get matches where bowler played
	bowler_matches = (
		bowler_data.groupby("match_id").agg(
			runs=("bowler_runs_conceded", "sum"),
			balls=("bowler_legal_ball", "sum"),
			wickets=("bowler_wicket", "sum"),
		)
	).reset_index()
	
	bowler_matches = bowler_matches.merge(match_info, on="match_id", how="left")
	bowler_matches = bowler_matches.sort_values(["date", "match_id"]).tail(5).copy()
	bowler_matches["date"] = bowler_matches["date"].dt.strftime("%Y-%m-%d")
	
	if len(bowler_matches) == 0:
		return {"games": [], "avg_wickets": 0, "avg_runs": 0, "avg_economy": 0}
	
	bowler_matches["economy"] = (
		bowler_matches["runs"] / (bowler_matches["balls"].replace(0, pd.NA) / 6)
	).fillna(0).round(2)
	
	return {
		"games": bowler_matches.to_dict("records"),
		"avg_wickets": bowler_matches["wickets"].mean(),
		"avg_runs": bowler_matches["runs"].mean(),
		"avg_economy": bowler_matches["economy"].mean(),
	}


@st.cache_data(show_spinner=False)
def get_head_to_head_stats(batter_name: str, bowler_name: str, all_deliveries: pd.DataFrame) -> dict:
	"""Get head-to-head statistics between batter and bowler"""
	h2h_data = all_deliveries[
		(all_deliveries["batter"] == batter_name) & (all_deliveries["bowler"] == bowler_name)
	].copy()
	h2h_data["is_legal_ball"] = h2h_data["extras_type"] != "wides"
	h2h_data["bowler_wicket"] = (
		(h2h_data["is_wicket"] == 1)
		& h2h_data["dismissal_kind"].isin(BOWLER_WICKET_KINDS)
	).astype(int)
	
	if len(h2h_data) == 0:
		return {
			"total_balls": 0,
			"total_runs": 0,
			"times_out": 0,
			"dismissal_kind": "N/A",
			"avg_runs_per_ball": 0,
			"wicket_probability": 0.05,
			"predicted_runs_next_over": 6.0,
		}
	
	total_balls = int(h2h_data["is_legal_ball"].sum())
	total_runs = h2h_data["batsman_runs"].sum()
	times_out = int(h2h_data["bowler_wicket"].sum())
	dismissal_kind = h2h_data[h2h_data["bowler_wicket"] == 1]["dismissal_kind"].values
	dismissal_kind = dismissal_kind[0] if len(dismissal_kind) > 0 else "N/A"
	
	avg_runs_per_ball = total_runs / total_balls if total_balls > 0 else 0
	wicket_probability = (times_out / total_balls * 100) if total_balls > 0 else 5.0
	predicted_runs_next_over = avg_runs_per_ball * 6
	
	return {
		"total_balls": total_balls,
		"total_runs": total_runs,
		"times_out": times_out,
		"dismissal_kind": dismissal_kind,
		"avg_runs_per_ball": round(avg_runs_per_ball, 2),
		"wicket_probability": round(wicket_probability, 2),
		"predicted_runs_next_over": round(predicted_runs_next_over, 2),
	}


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
	bowler: str,
	venue: str,
	player_data: pd.DataFrame,
	bowler_data: pd.DataFrame,
	all_deliveries: pd.DataFrame,
	match_info: pd.DataFrame,
) -> pd.DataFrame:
	balls_left = max(1, 120 - balls_bowled)
	runs_left = max(0, target_runs - current_runs)
	overs_completed = balls_bowled / 6

	crr = (current_runs * 6) / max(1, balls_bowled)
	rrr = (runs_left * 6) / balls_left

	# Get batter stats
	striker_rows = player_data[player_data["batter"] == striker]
	if striker_rows.empty:
		overall_runs = 0
		overall_balls = 0
		overall_sr = 0
	else:
		striker_stats = striker_rows.iloc[-1]
		overall_runs = float(striker_stats["overall_runs"])
		overall_balls = float(striker_stats["overall_balls"])
		overall_sr = float(striker_stats["overall_SR"])

	# Last-5 batter stats aligned with detailed section logic
	batter_last5 = get_batter_last5_games(striker, all_deliveries, match_info)
	if batter_last5["games"]:
		batter_last5_df = pd.DataFrame(batter_last5["games"])
		last5_runs = float(batter_last5_df["runs"].sum())
		last5_balls = float(batter_last5_df["balls"].sum())
		last5_sr = float(batter_last5["avg_sr"])
	else:
		last5_runs = 0.0
		last5_balls = 0.0
		last5_sr = 0.0

	# Get bowler stats
	bowler_stats_df = bowler_data[bowler_data["bowler"] == bowler]
	if len(bowler_stats_df) > 0:
		bowler_stat = bowler_stats_df.iloc[-1]
		overall_wickets = float(bowler_stat["overall_wickets"])
		overall_runs_conceded = float(bowler_stat["overall_runs_conceded"])
		overall_balls_bowled = float(bowler_stat["overall_balls_bowled"])
		bowling_avg = float(bowler_stat["bowling_avg"])
		bowling_economy = float(bowler_stat["bowling_economy"])
		last5_wickets = float(bowler_stat["last5_wickets"])
		last5_economy = float(bowler_stat["last5_economy"])
	else:
		# Default values if bowler not found
		overall_wickets = 0
		overall_runs_conceded = 0
		overall_balls_bowled = 0
		bowling_avg = 0
		bowling_economy = 0

	# Last-5 bowler stats aligned with detailed section logic
	bowler_last5 = get_bowler_last5_games(bowler, all_deliveries, match_info)
	if bowler_last5["games"]:
		bowler_last5_df = pd.DataFrame(bowler_last5["games"])
		last5_wickets = float(bowler_last5_df["wickets"].sum())
		last5_economy = float(bowler_last5["avg_economy"])
	else:
		last5_wickets = 0.0
		last5_economy = 0.0

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
		"overall_wickets": [overall_wickets],
		"overall_runs_conceded": [overall_runs_conceded],
		"overall_balls_bowled": [overall_balls_bowled],
		"bowling_avg": [bowling_avg],
		"bowling_economy": [bowling_economy],
		"last5_wickets": [last5_wickets],
		"last5_economy": [last5_economy],
		"current_phase": [phase_from_balls_left(balls_left)],
		"over_completed": [overs_completed],
	}
	return pd.DataFrame(match_data)


st.set_page_config(page_title="IPL Win Predictor", page_icon="cricket", layout="wide")

st.markdown("# IPL 2008-2024 Data")
st.markdown("## IPL Chase Win Probability Predictor")
st.caption("Enter the current chase situation to estimate winning chances for both teams.")

if not MODEL_PATH.exists():
	st.error("Model file not found: ipl_model.joblib")
	st.stop()

if not DELIVERIES_PATH.exists() or not MATCHES_PATH.exists():
	st.error("CSV files not found. Please keep deliveries.csv and matches.csv inside CSV/ folder.")
	st.stop()

with st.spinner("Loading model and reference data..."):
	model = load_model()
	venues, player_stats, team_batters_map, all_deliveries, bowler_stats, match_info = load_reference_data()

st.sidebar.header("Input Guide")
st.sidebar.write("- Use completed balls in 2nd innings (0 to 119).")
st.sidebar.write("- Choose striker from the batting team player list.")
st.sidebar.write("- App auto-computes phase, rates, and batter form features.")

# Inputs section
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

# Bowler selection
bowler_options_all = sorted(all_deliveries[all_deliveries["bowling_team"] == bowling_team]["bowler"].dropna().unique().tolist())
if bowler_options_all:
	bowler = st.selectbox("Select Bowler", options=bowler_options_all)
else:
	bowler = None
	st.warning("No bowler data available for the selected bowling team.")

predict_clicked = st.button("Predict Win Probability", type="primary", use_container_width=True)

# Prediction and Detailed Statistics Combined
if predict_clicked:
	st.markdown("### Win Probability Prediction")
	
	if target_runs <= current_runs:
		st.success("Batting team has already reached/passed the target. Win probability is 100%.")
		st.stop()

	if bowler is None:
		st.error("No bowler selected. Cannot make prediction.")
		st.stop()

	feature_row = build_match_features(
		batting_team=batting_team,
		bowling_team=bowling_team,
		balls_bowled=balls_bowled,
		current_runs=current_runs,
		target_runs=target_runs,
		wickets_left=wickets_left,
		striker=striker,
		bowler=bowler,
		venue=venue,
		player_data=player_stats,
		bowler_data=bowler_stats,
		all_deliveries=all_deliveries,
		match_info=match_info,
	)

	probabilities = predict_with_model(model, feature_row)
	bowling_win_pct = float(probabilities[0]) * 100
	batting_win_pct = float(probabilities[1]) * 100

	st.markdown("#### Predicted Winning Chances")
	metric_col1, metric_col2 = st.columns(2)
	metric_col1.metric(f"{batting_team}", f"{batting_win_pct:.2f}%")
	metric_col2.metric(f"{bowling_team}", f"{bowling_win_pct:.2f}%")

	st.progress(min(max(int(round(batting_win_pct)), 0), 100), text=f"{batting_team} win chance")
	st.progress(min(max(int(round(bowling_win_pct)), 0), 100), text=f"{bowling_team} win chance")

	with st.expander("Computed Match Snapshot - Model Features"):
		
		st.markdown("**Raw Feature DataFrame:**")
		st.dataframe(feature_row, use_container_width=True)

	st.markdown("---")
	st.markdown("### Detailed Player & Head-to-Head Statistics")
	
	if bowler is not None:
		# Batter Statistics
		st.markdown(f"#### {striker}'s Last 5 Games (Batter)")
		batter_stats = get_batter_last5_games(striker, all_deliveries, match_info)
		
		if batter_stats["games"]:
			batter_df = pd.DataFrame(batter_stats["games"])
			st.dataframe(batter_df, use_container_width=True)
			
			col1, col2, col3, col4 = st.columns(4)
			col1.metric("Average Runs", f"{batter_stats['avg_runs']:.2f}")
			col2.metric("Average Strike Rate", f"{batter_stats['avg_sr']:.2f}%")
			col3.metric("Fours (Last 5)", batter_stats["total_4s"])
			col4.metric("Sixes (Last 5)", batter_stats["total_6s"])
		else:
			st.info("No data available for this batter in last 5 games.")
		
		st.markdown("---")
		
		# Bowler Statistics
		st.markdown(f"#### {bowler}'s Last 5 Games (Bowler)")
		bowler_stats = get_bowler_last5_games(bowler, all_deliveries, match_info)
		
		if bowler_stats["games"]:
			bowler_df = pd.DataFrame(bowler_stats["games"])
			st.dataframe(bowler_df, use_container_width=True)
			
			col1, col2, col3 = st.columns(3)
			col1.metric("Average Wickets", f"{bowler_stats['avg_wickets']:.2f}")
			col2.metric("Average Runs Given", f"{bowler_stats['avg_runs']:.2f}")
			col3.metric("Average Economy", f"{bowler_stats['avg_economy']:.2f}")
		else:
			st.info("No data available for this bowler in last 5 games.")
		
		st.markdown("---")
		
		# Head-to-Head Statistics
		st.markdown(f"#### {striker} vs {bowler} - Head-to-Head")
		h2h_stats = get_head_to_head_stats(striker, bowler, all_deliveries)
		
		col1, col2, col3, col4 = st.columns(4)
		col1.metric("Total Balls Faced", h2h_stats["total_balls"])
		col2.metric("Total Runs", h2h_stats["total_runs"])
		col3.metric("Times Out", h2h_stats["times_out"])
		col4.metric("Dismissal Type", h2h_stats["dismissal_kind"])
		
		st.markdown("---")
		
		# Probability Predictions
		st.markdown("#### Match Probabilities")
		pred_col1, pred_col2 = st.columns(2)
		
		with pred_col1:
			st.metric(
				f"Wicket Probability (in next over)",
				f"{h2h_stats['wicket_probability']:.2f}%",
				delta="Higher = Increased risk"
			)
		
		with pred_col2:
			st.metric(
				f"Predicted Runs (next over)",
				f"{h2h_stats['predicted_runs_next_over']:.2f}",
				delta="Based on h2h runs/ball"
			)
