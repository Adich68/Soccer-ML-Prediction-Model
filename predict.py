import sys
import pandas as pd
import joblib
import numpy as np

# Load model + label encoder
model = joblib.load("models/epl_model.joblib")
le = joblib.load("models/label_encoder.joblib")

# Load dataset (for Elo reference)
df = pd.read_csv("data/epl_matches.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# === Recompute Elo ratings up to now ===
teams = pd.unique(df[['HomeTeam','AwayTeam']].values.ravel())
elo = {t: 1500 for t in teams}

for _, row in df.iterrows():
    Rh, Ra = elo[row['HomeTeam']], elo[row['AwayTeam']]
    exp_h = 1/(1+10**((Ra-Rh)/400))
    if row['FTR']=='H': res_h = 1
    elif row['FTR']=='D': res_h = 0.5
    else: res_h = 0
    elo[row['HomeTeam']] = Rh + 20*(res_h - exp_h)
    elo[row['AwayTeam']] = Ra + 20*((1-res_h) - (1-exp_h))

# === Handle CLI args ===
if len(sys.argv) != 3:
    print("Usage: python predict.py 'HomeTeam' 'AwayTeam'")
    sys.exit(1)

home, away = sys.argv[1], sys.argv[2]

# If new team not in dataset, assign baseline Elo
Rh = elo.get(home, 1500)
Ra = elo.get(away, 1500)

# Build feature row
features = pd.DataFrame([{
    "home_elo": Rh,
    "away_elo": Ra,
    "elo_diff": Rh - Ra,
    "home_adv": 1
}])

# Predict
probs = model.predict_proba(features)[0]
classes = le.inverse_transform(np.arange(len(probs)))

print(f"\nPrediction: {home} vs {away}")
for c, p in zip(classes, probs):
    print(f"{c.capitalize():5s}: {p*100:.2f}%")
