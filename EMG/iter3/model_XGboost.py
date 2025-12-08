import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import xgboost as xgb

# ===================== CONFIG =====================
FEATURE_CSV = "emg_features_3class_main.csv"
MODEL_PATH  = "emg_wrist_xgb_model.pkl"
LABEL_ENCODER_PATH = "emg_label_encoder.pkl"
# ==================================================


def main():
    # 1. Load feature data
    df = pd.read_csv(FEATURE_CSV)

    # Features used (must match your feature-extraction code)
    feature_cols = ["MAV", "RMS", "VAR", "WL", "ZC", "SSC", "WAMP", "IEMG"]

    X = df[feature_cols].values
    y_str = df["label"].values

    print("First few labels:", y_str[:10])

    # 2. Encode labels ("REST", "UP", "DOWN") -> integers 0..K-1
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes = le.classes_
    num_classes = len(classes)

    print("Classes:", classes)
    print("Number of classes:", num_classes)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # 4. Build XGBoost classifier
    # (Good starting hyperparameters, tune later if needed)
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",   # fast on CPU
        random_state=42,
    )

    # 5. Train with early stopping
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,           # set True if you want training logs
        # early_stopping_rounds=30
    )

    # 6. Evaluate
    y_pred = model.predict(X_test)

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=classes))

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print(confusion_matrix(y_test, y_pred))

    acc = np.mean(y_pred == y_test)
    print(f"\nOverall accuracy: {acc:.3f}")

    # 7. Feature importances
    print("\n=== Feature importances ===")
    for name, imp in zip(feature_cols, model.feature_importances_):
        print(f"{name:6s}: {imp:.3f}")

    # 8. Save model + label encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"\n✅ Saved XGBoost model to {MODEL_PATH}")
    print(f"✅ Saved label encoder to {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()