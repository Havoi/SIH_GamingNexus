import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ===================== CONFIG =====================
FEATURE_CSV = "emg_features_3class_main.csv"
MODEL_PATH  = "emg_wrist_model.pkl"
LABEL_ENCODER_PATH = "emg_label_encoder.pkl"
# ==================================================


def main():
    # 1. Load feature data
    df = pd.read_csv(FEATURE_CSV)

    # Columns we will use as features (same as we computed earlier)
    feature_cols = ["MAV", "RMS", "VAR", "WL", "ZC", "SSC", "WAMP", "IEMG"]

    # X = features, y = labels (REST / UP / DOWN)
    X = df[feature_cols].values
    y_str = df["label"].values

    print("First few labels:", y_str[:10])

    # 2. Encode labels to integers (REST/UP/DOWN -> 0/1/2 etc.)
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    print("Classes:", le.classes_)
    # Example mapping: index 0: DOWN, index 1: REST, index 2: UP

    # 3. Train / test split (stratified so label distribution is preserved)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Build model pipeline: StandardScaler + LogisticRegression
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            multi_class="multinomial"  # softmax for 3 classes
        )
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

    # 7. Save model + label encoder for later real-time use
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"\n✅ Saved model to {MODEL_PATH}")
    print(f"✅ Saved label encoder to {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()