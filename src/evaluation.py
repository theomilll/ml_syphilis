# evaluation.py
# Contains functions for model evaluation, including metrics, plots, and saving results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    RocCurveDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
    average_precision_score
)
import joblib
import os

# Seed for reproducibility as per project guidelines
import random
random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# --- Evaluation Functions ---

def generate_and_save_classification_report(y_true, y_pred, model_name, tables_dir):
    """
    Generates, prints, and saves a classification report to a CSV file.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name (str): Name of the model for titling and filenames.
        tables_dir (str): Directory to save the report CSV.

    Returns:
        pd.DataFrame: The classification report as a DataFrame.
    """
    os.makedirs(tables_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    
    print(f"\nClassification Report for {model_name}:\n")
    print(report_df)
    
    report_filename = os.path.join(tables_dir, f"{model_name}_classification_report.csv")
    report_df.to_csv(report_filename)
    print(f"Classification report saved to: {report_filename}")
    return report_df

def plot_and_save_confusion_matrix(y_true, y_pred, model_name, figures_dir, display_labels=None):
    """
    Plots and saves a confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name (str): Name of the model for titling and filenames.
        figures_dir (str): Directory to save the plot.
        display_labels (list, optional): Labels for the matrix axes. Defaults to None.

    Returns:
        np.ndarray: The confusion matrix.
    """
    os.makedirs(figures_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d') # 'd' for integer format
    ax.set_title(f"Confusion Matrix: {model_name}")
    
    cm_filename = os.path.join(figures_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_filename)
    print(f"Confusion matrix plot saved to: {cm_filename}")
    plt.show()
    return cm

def plot_and_save_roc_curve(estimator, X_test, y_test, model_name, figures_dir):
    """
    Plots and saves the ROC curve for a given estimator.

    Args:
        estimator: Trained model with a predict_proba method.
        X_test: Test features.
        y_test: True test labels.
        model_name (str): Name of the model for titling and filenames.
        figures_dir (str): Directory to save the plot.

    Returns:
        float: The Area Under the ROC Curve (AUC).
    """
    os.makedirs(figures_dir, exist_ok=True)
    y_pred_proba = estimator.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)') # Add chance line
    ax.set_title(f"ROC Curve: {model_name} (AUC = {roc_auc:.2f})")
    ax.legend()

    roc_filename = os.path.join(figures_dir, f"{model_name}_roc_curve.png")
    plt.savefig(roc_filename)
    print(f"ROC curve plot saved to: {roc_filename}")
    plt.show()
    return roc_auc

def plot_and_save_pr_curve(estimator, X_test, y_test, model_name, figures_dir):
    """
    Plots and saves the Precision-Recall curve for a given estimator.

    Args:
        estimator: Trained model with a predict_proba method.
        X_test: Test features.
        y_test: True test labels.
        model_name (str): Name of the model for titling and filenames.
        figures_dir (str): Directory to save the plot.

    Returns:
        float: The Average Precision (AP) score.
    """
    os.makedirs(figures_dir, exist_ok=True)
    y_pred_proba = estimator.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=avg_precision, estimator_name=model_name)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    ax.set_title(f"Precision-Recall Curve: {model_name} (AP = {avg_precision:.2f})")
    
    pr_filename = os.path.join(figures_dir, f"{model_name}_pr_curve.png")
    plt.savefig(pr_filename)
    print(f"Precision-Recall curve plot saved to: {pr_filename}")
    plt.show()
    return avg_precision

def run_model_evaluation(model, X_test, y_test, model_name_prefix, figures_dir, tables_dir, class_labels=['Negative (0)', 'Positive (1)'], **kwargs):
    """
    Orchestrates the evaluation of a single model on the test set.

    Args:
        model: Trained model object.
        X_test: Test features.
        y_test: True test labels.
        model_name_prefix (str): Prefix for filenames (e.g., 'logistic_regression').
        figures_dir (str): Directory to save plots.
        tables_dir (str): Directory to save report CSVs.
        class_labels (list, optional): Labels for confusion matrix. 
                                      Defaults to ['Negative (0)', 'Positive (1)'].
    """
    print(f"\n{'='*20} Evaluating {model_name_prefix} {'='*20}")
    
    # Ensure output directories exist
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Get predictions
    # If a specific threshold is provided, use it with predict_proba
    # Otherwise, use the default model.predict() which typically uses a 0.5 threshold
    custom_threshold = kwargs.get('threshold', None)
    if custom_threshold is not None and hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for positive class
        y_pred = (y_pred_proba >= custom_threshold).astype(int)
        print(f"Evaluating with custom threshold: {custom_threshold}")
    else:
        y_pred = model.predict(X_test)
        if custom_threshold is not None:
            print(f"Warning: Threshold provided but model {model_name_prefix} may not support predict_proba or it wasn't used.")

    # 1. Classification Report
    print("\n--- Generating Classification Report ---")
    generate_and_save_classification_report(y_test, y_pred, model_name_prefix, tables_dir)

    # 2. Confusion Matrix
    print("\n--- Plotting Confusion Matrix ---")
    plot_and_save_confusion_matrix(y_test, y_pred, model_name_prefix, figures_dir, display_labels=class_labels)

    # 3. ROC Curve (requires predict_proba)
    if hasattr(model, "predict_proba"):
        print("\n--- Plotting ROC Curve ---")
        plot_and_save_roc_curve(model, X_test, y_test, model_name_prefix, figures_dir)
    else:
        print(f"Skipping ROC curve for {model_name_prefix} as it does not have 'predict_proba' method.")

    # 4. Precision-Recall Curve (requires predict_proba)
    if hasattr(model, "predict_proba"):
        print("\n--- Plotting Precision-Recall Curve ---")
        plot_and_save_pr_curve(model, X_test, y_test, model_name_prefix, figures_dir)
    else:
        print(f"Skipping Precision-Recall curve for {model_name_prefix} as it does not have 'predict_proba' method.")
        
    print(f"{'='*20} Evaluation for {model_name_prefix} completed {'='*20}")

def evaluate_with_threshold_tuning(model, X_test, y_test, model_name_prefix, figures_dir, tables_dir, class_labels=['Negative (0)', 'Positive (1)'], positive_class_label=1):
    """
    Evaluates a model by tuning the prediction threshold, plots metrics vs. threshold,
    and allows re-evaluation with a chosen or optimal threshold.

    Args:
        model: Trained model object with predict_proba method.
        X_test: Test features.
        y_test: True test labels.
        model_name_prefix (str): Prefix for filenames.
        figures_dir (str): Directory to save plots.
        tables_dir (str): Directory to save report CSVs.
        class_labels (list): Labels for confusion matrix.
        positive_class_label (int): The label of the positive class (e.g., 1).

    Returns:
        pd.DataFrame: DataFrame containing metrics for different thresholds.
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    if not hasattr(model, "predict_proba"):
        print(f"Model {model_name_prefix} does not have predict_proba method. Skipping threshold tuning.")
        return None

    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

    thresholds = np.arange(0.01, 1.0, 0.01)
    metrics_list = []

    for thresh in thresholds:
        y_pred_tuned = (y_pred_proba >= thresh).astype(int)
        report = classification_report(y_test, y_pred_tuned, output_dict=True, zero_division=0)
        metrics_list.append({
            'threshold': thresh,
            'precision_pos': report[str(positive_class_label)]['precision'],
            'recall_pos': report[str(positive_class_label)]['recall'],
            'f1_pos': report[str(positive_class_label)]['f1-score'],
            'precision_neg': report[str(1-positive_class_label)]['precision'],
            'recall_neg': report[str(1-positive_class_label)]['recall'],
            'accuracy': report['accuracy']
        })
    
    metrics_df = pd.DataFrame(metrics_list)

    # Plotting metrics vs. threshold
    plt.figure(figsize=(12, 7))
    plt.plot(metrics_df['threshold'], metrics_df['precision_pos'], label='Precision (Positive Class)')
    plt.plot(metrics_df['threshold'], metrics_df['recall_pos'], label='Recall (Positive Class)', linewidth=2)
    plt.plot(metrics_df['threshold'], metrics_df['f1_pos'], label='F1-score (Positive Class)')
    plt.plot(metrics_df['threshold'], metrics_df['accuracy'], label='Accuracy', linestyle='--')
    plt.title(f'Metrics vs. Prediction Threshold for {model_name_prefix}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(figures_dir, f"{model_name_prefix}_metrics_vs_threshold.png")
    plt.savefig(plot_filename)
    print(f"Metrics vs. threshold plot saved to: {plot_filename}")
    plt.show()

    # Example: Find threshold that maximizes F1-score for the positive class
    # optimal_idx_f1 = metrics_df['f1_pos'].idxmax()
    # optimal_threshold_f1 = metrics_df.loc[optimal_idx_f1, 'threshold']
    # print(f"\nThreshold maximizing F1-score for positive class: {optimal_threshold_f1:.2f}")
    # print(f"Metrics at this threshold (F1-max):\n{metrics_df.loc[optimal_idx_f1]}")

    # Example: Find threshold that maximizes recall for the positive class (with some constraints if needed)
    # optimal_idx_recall = metrics_df['recall_pos'].idxmax()
    # optimal_threshold_recall = metrics_df.loc[optimal_idx_recall, 'threshold']
    # print(f"\nThreshold maximizing Recall for positive class: {optimal_threshold_recall:.2f}")
    # print(f"Metrics at this threshold (Recall-max):\n{metrics_df.loc[optimal_idx_recall]}")
    
    # User should inspect the plot and metrics_df to choose a threshold.
    # For now, we just return the metrics_df. The actual re-evaluation with a chosen threshold
    # will be done in the notebook after inspecting these results.
    
    return metrics_df

print("evaluation.py loaded with comprehensive model evaluation and threshold tuning functions.")
