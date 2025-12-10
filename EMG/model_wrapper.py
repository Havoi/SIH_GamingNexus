"""
model_wrapper.py

Unified model loader and inference helper.
Supports classifiers with `predict_proba` (e.g., RandomForest, MLP) and models which only
implement `predict`. Optional scaler and label encoder support included.

Primary class: ModelWrapper
- Methods: predict_proba_active(feats_2d), predict_label(feats_2d), reload()

"""
from __future__ import annotations
import os
import joblib
import numpy as np
from typing import Optional


class ModelWrapper:
    def __init__(self, model_path: str, scaler_path: Optional[str] = None,
                 le_path: Optional[str] = None, active_label: str = 'ACTIVE'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.le_path = le_path
        self.active_label = active_label

        self.model = None
        self.scaler = None
        self.le = None
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

        if self.scaler_path and os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
            except Exception:
                self.scaler = None

        if self.le_path and os.path.exists(self.le_path):
            try:
                self.le = joblib.load(self.le_path)
            except Exception:
                self.le = None

    def transform_feats(self, feats_2d: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            try:
                return self.scaler.transform(feats_2d)
            except Exception:
                return feats_2d
        return feats_2d

    def predict_proba_active(self, feats_2d: np.ndarray) -> float:
        """Return probability for the ACTIVE class for a single row of features.
        feats_2d: shape (1, n_features)
        """
        feats_in = self.transform_feats(feats_2d)

        # try predict_proba first
        if hasattr(self.model, 'predict_proba'):
            try:
                probs = self.model.predict_proba(feats_in)
                classes = None
                if self.le is not None and hasattr(self.le, 'classes_'):
                    classes = self.le.classes_
                elif hasattr(self.model, 'classes_'):
                    classes = self.model.classes_

                if classes is not None:
                    # try to find index of ACTIVE label (case-insensitive match)
                    idx = None
                    for i, c in enumerate(classes):
                        try:
                            if str(c).strip().upper() == self.active_label.strip().upper():
                                idx = i
                                break
                        except Exception:
                            continue
                    if idx is not None:
                        return float(probs[0, idx])
                    # fallback heuristic for binary models
                    if probs.shape[1] == 2:
                        return float(probs[0, 1])
                # last resort: return max class probability
                return float(np.max(probs[0]))
            except Exception:
                pass

        # fallback: predict and map to binary 0/1
        try:
            pred = self.model.predict(feats_in)
            p = pred[0] if isinstance(pred, (list, tuple, np.ndarray)) else pred

            # if label encoder exists, invert and compare
            if self.le is not None:
                try:
                    lbl = str(self.le.inverse_transform([p])[0]).strip().upper()
                    return 1.0 if lbl == self.active_label.strip().upper() else 0.0
                except Exception:
                    pass

            # if prediction is a string label
            if isinstance(p, str):
                return 1.0 if p.strip().upper() == self.active_label.strip().upper() else 0.0

            # if prediction is integer index into classes_
            if hasattr(self.model, 'classes_'):
                classes = self.model.classes_
                try:
                    if isinstance(p, (int, np.integer)) and 0 <= int(p) < len(classes):
                        lbl = str(classes[int(p)]).strip().upper()
                        return 1.0 if lbl == self.active_label.strip().upper() else 0.0
                except Exception:
                    pass
        except Exception:
            pass

        return 0.0

    def predict_label(self, feats_2d: np.ndarray):
        feats_in = self.transform_feats(feats_2d)
        try:
            pred = self.model.predict(feats_in)
            if isinstance(pred, (list, tuple, np.ndarray)):
                return pred[0]
            return pred
        except Exception:
            return None

    def reload(self):
        """Reload model, scaler and label encoder from disk. Useful for hot-reload in GUI."""
        self._load()


# small self-test when executed
if __name__ == '__main__':
    print('ModelWrapper self-test: create wrapper with non-existing path to see exception handling')
    try:
        mw = ModelWrapper('nonexistent.joblib')
    except Exception as e:
        print('Expected error:', e)
