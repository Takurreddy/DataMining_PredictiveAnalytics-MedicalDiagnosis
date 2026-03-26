"""
utils/evaluator.py
------------------
Common evaluation functions for all classifiers:
  - 10-fold stratified cross-validation
  - Accuracy, Kappa, Precision, Recall, F1-score
  - Confusion matrix plot
  - Model comparison bar chart
"""

import os
import warnings
import numpy as np
warnings.filterwarnings('ignore', message='.*number of unique classes.*')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def cross_validate_model(model, X, y, n_folds: int = 10, random_state: int = 42):
    """
    Run k-fold cross-validation. Automatically reduces n_folds
    if the dataset is too small for stratified splitting.

    Returns
    -------
    dict with keys: mean_accuracy, std_accuracy, all_scores, n_folds
    """
    # Determine safe number of folds based on minimum class count
    min_class_count = np.bincount(y).min() if len(np.unique(y)) > 1 else 1
    safe_folds = min(n_folds, min_class_count, len(X))
    if safe_folds < 2:
        safe_folds = 2

    # Use KFold when classes are too sparse for stratification
    if min_class_count < safe_folds:
        cv = KFold(n_splits=safe_folds, shuffle=True, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=random_state)

    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    return {
        'mean_accuracy': scores.mean(),
        'std_accuracy':  scores.std(),
        'all_scores':    scores,
        'n_folds':       safe_folds,
    }


def evaluate_model(model, X_train, X_test, y_train, y_test, le, model_name: str):
    """
    Fit model, evaluate on test set, print full report, and save confusion matrix.

    Returns
    -------
    dict of metrics
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    kappa  = cohen_kappa_score(y_test, y_pred)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

    # ── Console output ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Kappa     : {kappa:.4f}")
    print(f"  MAE       : {mae:.4f}")
    print(f"  RMSE      : {rmse:.4f}")
    print(f"\n  Classification Report:")
    target_names = le.inverse_transform(np.unique(y_test))
    print(classification_report(
        y_test, y_pred,
        labels=np.unique(y_test),
        target_names=target_names,
        zero_division=0
    ))

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    _save_confusion_matrix(y_test, y_pred, le, model_name)

    return {
        'model_name': model_name,
        'accuracy':   acc,
        'kappa':      kappa,
        'mae':        mae,
        'rmse':       rmse,
    }


def _save_confusion_matrix(y_test, y_pred, le, model_name: str):
    """Save a heatmap of the confusion matrix to results/."""
    labels = np.unique(np.concatenate([y_test, y_pred]))
    class_names = le.inverse_transform(labels)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(max(10, len(class_names)), max(8, len(class_names) - 2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    filename = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {filename}")


def plot_model_comparison(results: list):
    """
    Bar chart comparing accuracy of all models.

    Parameters
    ----------
    results : list of dicts, each with 'model_name' and 'accuracy'
    """
    names     = [r['model_name'] for r in results]
    accs      = [r['accuracy'] * 100 for r in results]
    kappas    = [r['kappa'] for r in results]

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy bar chart
    bars = axes[0].bar(names, accs, color=colors, edgecolor='black', linewidth=0.8)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xticklabels(names, rotation=15, ha='right')
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Kappa bar chart
    bars2 = axes[1].bar(names, kappas, color=colors, edgecolor='black', linewidth=0.8)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel('Kappa Statistic', fontsize=12)
    axes[1].set_title('Kappa Statistic Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xticklabels(names, rotation=15, ha='right')
    for bar, k in zip(bars2, kappas):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{k:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Predictive Analytics — Algorithm Performance Comparison',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    filename = os.path.join(RESULTS_DIR, 'model_comparison.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Model comparison chart saved → {filename}")


def print_summary_table(results: list):
    """Print a formatted summary table of all model results."""
    print("\n" + "=" * 70)
    print("  FINAL MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Algorithm':<25} {'Accuracy':>10} {'Kappa':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 70)

    best = max(results, key=lambda r: r['accuracy'])
    for r in sorted(results, key=lambda r: r['accuracy'], reverse=True):
        marker = " ◀ BEST" if r['model_name'] == best['model_name'] else ""
        print(f"  {r['model_name']:<25} {r['accuracy']*100:>9.2f}% "
              f"{r['kappa']:>10.4f} {r['mae']:>10.4f} {r['rmse']:>10.4f}{marker}")
    print("=" * 70)
