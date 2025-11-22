import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ===================== CONFIG =====================
FEATURE_CSV = "emg_features_3class.csv"
MODEL_PATH  = "emg_wrist_rf_model.pkl"
LABEL_ENCODER_PATH = "emg_label_encoder.pkl"
# ==================================================


def main():
    # 1. Load feature data
    df = pd.read_csv(FEATURE_CSV)

    # Features we computed earlier
    feature_cols = ["MAV", "RMS", "VAR", "WL", "ZC", "SSC", "WAMP", "IEMG"]

    X = df[feature_cols].values
    y_str = df["label"].values

    print("First few labels:", y_str[:10])

    # 2. Encode text labels ("REST", "UP", "DOWN") -> integers
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    print("Classes:", le.classes_)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Build model: (optional) StandardScaler + RandomForest
    # RF doesn't strictly need scaling, but using a pipeline keeps code consistent
    rf = RandomForestClassifier(
        n_estimators=200,      # number of trees
        max_depth=None,       # allow full depth (you can limit to reduce overfitting)
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",  # helps with REST/UP/DOWN imbalance
        random_state=42,
        n_jobs=-1             # use all CPU cores
    )

    model = make_pipeline(
        StandardScaler(),
        rf
    )

    # 5. Train
    model.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test)

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print(confusion_matrix(y_test, y_pred))

    acc = np.mean(y_pred == y_test)
    print(f"\nOverall accuracy: {acc:.3f}")

    # 7. Show feature importances (from inside the pipeline)
    rf_model: RandomForestClassifier = model.named_steps[
        [k for k in model.named_steps if isinstance(model.named_steps[k], RandomForestClassifier)][0]
    ]
    print("\n=== Feature importances ===")
    for name, imp in zip(feature_cols, rf_model.feature_importances_):
        print(f"{name:6s}: {imp:.3f}")

    # 8. Save model + label encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"\n✅ Saved Random Forest model to {MODEL_PATH}")
    print(f"✅ Saved label encoder to {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()