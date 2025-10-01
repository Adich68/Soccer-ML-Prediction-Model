import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

# === 1. Load data ===
# Example: Football-Data.co.uk CSVs have columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
df = pd.read_csv("data/epl_matches.csv", parse_dates=['Date'])
df = df.sort_values("Date").reset_index(drop=True)

# === 2. Compute simple Elo ratings ===
teams = pd.unique(df[['HomeTeam','AwayTeam']].values.ravel())
elo = {t: 1500 for t in teams}
elos_home, elos_away = [], []
for _, row in df.iterrows():
    Rh, Ra = elo[row['HomeTeam']], elo[row['AwayTeam']]
    exp_h = 1/(1+10**((Ra-Rh)/400))
    if row['FTR']=='H': res_h = 1
    elif row['FTR']=='D': res_h = 0.5
    else: res_h = 0
    elo[row['HomeTeam']] = Rh + 20*(res_h - exp_h)
    elo[row['AwayTeam']] = Ra + 20*((1-res_h) - (1-exp_h))
    elos_home.append(Rh)
    elos_away.append(Ra)

df['home_elo'] = elos_home
df['away_elo'] = elos_away
df['elo_diff'] = df['home_elo'] - df['away_elo']
df['home_adv'] = 1
df['target'] = df['FTR'].map({'H':'home','D':'draw','A':'away'})

# === 3. Features & target ===
FEATURES = ['home_elo','away_elo','elo_diff','home_adv']
X = df[FEATURES]
y = df['target']

# === 4. Train/test split (time-based, no shuffle) ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === 5. Train model ===
clf = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", use_label_encoder=False)
clf.fit(X_train, y_train)

# Calibrate probabilities
calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
calibrated.fit(X_test, y_test)

# === 6. Evaluate ===
y_pred = calibrated.predict(X_test)
y_prob = calibrated.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))

# === 7. Save model ===
Path("models").mkdir(exist_ok=True)
joblib.dump(calibrated, "models/epl_model.joblib")
print("✅ Model saved to models/epl_model.joblib")
