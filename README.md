# Premier League Match Outcome Prediction (Machine Learning)

This project builds a machine learning model to predict **English Premier League match outcomes** (Home Win / Draw / Away Win).  
It uses **Elo ratings** and simple feature engineering, then trains an **XGBoost classifier** to generate calibrated probabilities.  

## Features
- Historical EPL match data (Football-Data.co.uk / Kaggle datasets)
- Feature engineering: Elo ratings, home advantage, team form
- ML model: XGBoost + probability calibration
- Evaluation: Accuracy + Log Loss
- Saves trained model as `models/epl_model.joblib`

## Project Structure
```
epl-prediction-model/
├─ data/            # place your CSV here (e.g. Football-Data EPL matches)
├─ models/          # trained model saved here
├─ epl_model.py     # training script
├─ requirements.txt # dependencies
└─ README.md
```

## Setup
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/epl-prediction-model.git
cd epl-prediction-model

# Create venv
python3 -m venv .venv
source .venv/bin/activate  # (on Windows: .venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt

# Place dataset
mkdir data
# e.g. data/epl_matches.csv

# Run training
python epl_model.py
```

## Example Output
```
Accuracy: 0.59
Log Loss: 0.93
Model saved to models/epl_model.joblib
```

