"""
models/svm_classifier.py
------------------------
Support Vector Machine (SVM) classifier for disease prediction.

SVM finds the optimal hyperplane that maximises the margin between
disease classes in high-dimensional symptom space. The kernel trick
allows non-linear boundaries.

Equivalent to the WEKA configuration:
  SMO (Sequential Minimal Optimisation) — PolyKernel, normalised data

This model achieved 76.64% accuracy in the project.

Algorithm highlights:
  - kernel='rbf'  → Radial Basis Function handles non-linear symptom patterns
  - C=1.0         → regularisation (trade-off between margin and misclassification)
  - gamma='scale' → 1 / (n_features * X.var()), matches WEKA normalisation
"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_svm_model(
    C: float = 1.0,
    kernel: str = 'rbf',
    gamma: str = 'scale',
    random_state: int = 42
):
    """
    Build and return an SVM pipeline with feature scaling.

    SVM is sensitive to feature scale, so StandardScaler is included
    in the pipeline to normalise features before training — equivalent
    to WEKA's normalisation step in SMO.

    Parameters
    ----------
    C           : float — regularisation (lower = wider margin, more errors allowed)
    kernel      : str   — 'rbf', 'poly', or 'linear'
    gamma       : str   — kernel coefficient ('scale' or 'auto')
    random_state: int   — random seed for reproducibility

    Returns
    -------
    sklearn Pipeline with StandardScaler + SVC
    """
    model = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,    # enable predict_proba() for confidence scores
            random_state=random_state
        ))
    ])
    return model


def build_linear_svm_model(C: float = 1.0, random_state: int = 42):
    """
    Build a linear SVM — equivalent to WEKA's SMO with PolyKernel (exponent=1).

    Linear SVM is faster and interpretable for high-dimensional binary features.
    """
    model = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', SVC(
            C=C,
            kernel='linear',
            probability=True,
            random_state=random_state
        ))
    ])
    return model


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from utils.preprocessor import get_preprocessed_data
    from utils.evaluator import evaluate_model, cross_validate_model
    import numpy as np

    print("=" * 60)
    print("  SVM (SMO) — Disease Prediction")
    print("=" * 60)

    X_train, X_test, y_train, y_test, le, all_symptoms = get_preprocessed_data()

    # ── RBF SVM ──────────────────────────────────────────────────────────────
    model = build_svm_model()

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    cv_results = cross_validate_model(model, X_all, y_all, n_folds=10)
    print(f"\n  10-Fold CV Accuracy (RBF): {cv_results['mean_accuracy']*100:.2f}% "
          f"(± {cv_results['std_accuracy']*100:.2f}%)")

    evaluate_model(model, X_train, X_test, y_train, y_test, le, 'SVM (RBF Kernel)')

    # ── Linear SVM (closer to WEKA SMO) ──────────────────────────────────────
    linear_model = build_linear_svm_model()
    cv_lin = cross_validate_model(linear_model, X_all, y_all, n_folds=10)
    print(f"\n  10-Fold CV Accuracy (Linear): {cv_lin['mean_accuracy']*100:.2f}% "
          f"(± {cv_lin['std_accuracy']*100:.2f}%)")

    evaluate_model(linear_model, X_train, X_test, y_train, y_test, le, 'SVM (Linear Kernel)')
