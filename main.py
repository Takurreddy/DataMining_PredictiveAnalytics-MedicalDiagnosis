"""
main.py
-------
Runs all four classification algorithms on the disease-prediction dataset,
evaluates them using 10-fold cross-validation, prints a comparison table,
and saves charts to the results/ folder.

Usage:
    python main.py
"""

import sys
import os
import warnings
import numpy as np
warnings.filterwarnings('ignore', message='.*number of unique classes.*')

# ── Add project root to path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessor import get_preprocessed_data
from utils.evaluator import (
    evaluate_model,
    cross_validate_model,
    plot_model_comparison,
    print_summary_table
)
from models.naive_bayes    import build_naive_bayes_model
from models.random_forest  import build_random_forest_model, get_feature_importances
from models.svm_classifier import build_svm_model
from models.decision_tree  import build_decision_tree_model, print_tree_stats


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║      PREDICTIVE ANALYTICS IN MEDICAL DIAGNOSIS               ║
║      Aditya University — AIML Department — 2026              ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # ── Load & Preprocess Data ────────────────────────────────────────────────
    print("  [1/6]  Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test, le, all_symptoms = get_preprocessed_data(
        test_size=0.2, random_state=42
    )

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    print(f"         Training samples : {len(X_train)}")
    print(f"         Test samples     : {len(X_test)}")
    print(f"         Total symptoms   : {len(all_symptoms)}")
    print(f"         Disease classes  : {len(le.classes_)}")

    # ── Define Models ─────────────────────────────────────────────────────────
    models = [
        ('Naive Bayes',       build_naive_bayes_model()),
        ('Random Forest',     build_random_forest_model()),
        ('SVM (RBF)',         build_svm_model()),
        ('Decision Tree',     build_decision_tree_model()),
    ]

    all_results = []

    # ── Run 10-Fold CV + Full Evaluation for each model ───────────────────────
    total = len(models)
    for i, (name, model) in enumerate(models, start=2):
        print(f"\n  [{i}/{total+1}]  Running {name} ...")

        # 10-Fold Cross Validation
        cv = cross_validate_model(model, X_all, y_all, n_folds=10)
        print(f"         10-Fold CV: {cv['mean_accuracy']*100:.2f}% "
              f"(± {cv['std_accuracy']*100:.2f}%)")

        # Full train/test evaluation
        result = evaluate_model(model, X_train, X_test, y_train, y_test, le, name)
        result['cv_accuracy'] = cv['mean_accuracy']
        all_results.append(result)

    # ── Extra: Random Forest feature importances ──────────────────────────────
    rf_model = next(m for n, m in models if n == 'Random Forest')
    get_feature_importances(rf_model, all_symptoms, top_n=10)

    # ── Extra: Decision Tree stats ────────────────────────────────────────────
    dt_model = next(m for n, m in models if n == 'Decision Tree')
    print_tree_stats(dt_model)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary_table(all_results)
    plot_model_comparison(all_results)

    best = max(all_results, key=lambda r: r['accuracy'])
    print(f"\n  ✅  Best Algorithm: {best['model_name']} "
          f"({best['accuracy']*100:.2f}% accuracy)\n")


if __name__ == '__main__':
    main()
