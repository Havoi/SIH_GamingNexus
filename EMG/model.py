import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ======================================================
# CONFIG
# ======================================================
CSV_PATH = "emg_features_combined_2class_anirudh_randomnoise.csv"

MODEL_OUT = "emg_mlp_model_clean_radnomextnoisy.joblib"
SCALER_OUT = "emg_scaler_clean_randomextnoisy.pkl"
ENCODER_OUT = "emg_label_encoder_clean_randomextnoisy.pkl"

FEATURE_COLUMNS = [
    "MAV", "RMS", "VAR", "WL",
    "ZC", "SSC", "WAMP", "IEMG"
]

LABEL_COLUMN = "label"
# ======================================================


print("Loading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# Extract features and labels
X = df[FEATURE_COLUMNS].values
y_raw = df[LABEL_COLUMN].astype(str).values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)

print("Classes:", le.classes_)

# Scale features (VERY important for MLP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Build MLP model
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    max_iter=500,
    random_state=42
)

print("Training MLP...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nClassification report: \n", classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Save trained components
joblib.dump(model, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
joblib.dump(le, ENCODER_OUT)

print("\nSaved:")
print(" - Model:", MODEL_OUT)
print(" - Scaler:", SCALER_OUT)
print(" - Encoder:", ENCODER_OUT)