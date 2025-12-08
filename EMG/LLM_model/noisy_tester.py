import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

CSV_PATH = "emg_features_combined_2class_noisy.csv"
MODEL_OUT = "emg_mlp_model_clean_noisy.joblib"
SCALER_OUT = "emg_scaler_clean_noisy.pkl"
ENCODER_OUT = "emg_label_encoder_clean_noisy.pkl"

# load files
model = joblib.load(MODEL_OUT)
scaler = joblib.load(SCALER_OUT)
le = joblib.load(ENCODER_OUT)

df = pd.read_csv(CSV_PATH)

FEATURE_COLUMNS = ["MAV","RMS","VAR","WL","ZC","SSC","WAMP","IEMG"]
label_col = "label"

# Sanity: check column order present
assert all(c in df.columns for c in FEATURE_COLUMNS), "Missing feature columns in CSV!"

X = df[FEATURE_COLUMNS].values
y_raw = df[label_col].astype(str).values
y_true = le.transform(y_raw)   # encode with the same encoder

# Transform using the SAVED scaler (important)
X_scaled = scaler.transform(X)   # DO NOT fit again

# Test full dataset
y_pred = model.predict(X_scaled)
print("Accuracy on CSV (using saved scaler+model):", accuracy_score(y_true, y_pred))
print("Sample inverse labels:", le.inverse_transform(y_pred[:10]))
# Show a few example scaled vectors and predicted probs
print("Example scaled vector (first):", X_scaled[0])
print("Pred proba (first):", model.predict_proba(X_scaled[:5]))
