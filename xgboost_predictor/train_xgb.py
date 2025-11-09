#!/usr/bin/env python3
"""
Train XGBoost model to predict 1:2 risk-reward trade opportunities.
"""

import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib


def plot_feature_importance(model, feature_names, save_path):
    """Plot and save feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path):
    """Plot and save precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Precision-recall curve saved to: {save_path}")
    plt.close()


class LearningRateDecayCallback(xgb.callback.TrainingCallback):
    """Custom callback for learning rate decay in XGBoost."""

    def __init__(self, lr_start: float, lr_end: float, n_estimators: int):
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.n_estimators = n_estimators
        super().__init__()

    def after_iteration(self, model, epoch, evals_log):
        """Called after each boosting iteration."""
        # Linear decay from lr_start to lr_end
        progress = epoch / self.n_estimators
        new_lr = self.lr_start - (self.lr_start - self.lr_end) * progress

        # Update learning rate
        model.set_param({'learning_rate': new_lr})

        return False  # Continue training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="xgboost_predictor/labeled_data.csv",
                       help="Labeled data CSV")
    parser.add_argument("--model-dir", default="xgboost_predictor/models",
                       help="Directory to save trained model")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size (default: 0.2)")
    parser.add_argument("--time-series-split", action="store_true",
                       help="Use time-series split (train on early data, test on later data)")
    parser.add_argument("--max-depth", type=int, default=6,
                       help="Max tree depth (default: 6)")
    parser.add_argument("--n-estimators", type=int, default=100,
                       help="Number of boosting rounds (default: 100)")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                       help="Learning rate (default: 0.1)")
    parser.add_argument("--lr-decay", action="store_true",
                       help="Enable learning rate decay")
    parser.add_argument("--lr-end", type=float, default=0.01,
                       help="Final learning rate for decay (default: 0.01)")
    parser.add_argument("--early-stopping-rounds", type=int, default=None,
                       help="Stop if no improvement for N rounds (default: None)")
    parser.add_argument("--scale-pos-weight", type=float, default=None,
                       help="Scale positive class weight (auto if None)")
    parser.add_argument("--min-child-weight", type=int, default=1,
                       help="Minimum sum of instance weight in a child (default: 1)")
    parser.add_argument("--subsample", type=float, default=0.8,
                       help="Subsample ratio (default: 0.8)")
    parser.add_argument("--colsample-bytree", type=float, default=0.8,
                       help="Column subsample ratio (default: 0.8)")
    args = parser.parse_args()

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Total samples: {len(df):,}")

    # Filter out samples with no outcome (bars_to_outcome == -1)
    df_with_outcome = df[df['bars_to_outcome'] > 0].copy()
    print(f"Samples with outcome: {len(df_with_outcome):,}")

    # Prepare features and labels
    label_cols = ['target_hit', 'stop_hit', 'bars_to_outcome', 'entry_price', 'stop_price', 'target_price']
    feature_cols = [col for col in df_with_outcome.columns if col not in label_cols]

    X = df_with_outcome[feature_cols].values
    y = df_with_outcome['target_hit'].values

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Positive samples (target hit): {y.sum():,} ({y.sum()/len(y)*100:.1f}%)")
    print(f"Negative samples (stop hit): {len(y) - y.sum():,} ({(len(y) - y.sum())/len(y)*100:.1f}%)")

    # Calculate scale_pos_weight if not provided
    if args.scale_pos_weight is None:
        neg_count = len(y) - y.sum()
        pos_count = y.sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        print(f"Auto scale_pos_weight: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = args.scale_pos_weight

    # Split data
    if args.time_series_split:
        # Time-series split: train on early data, test on later data
        split_idx = int(len(X) * (1 - args.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        print(f"\nUsing TIME-SERIES split (train on early data, test on later data)")
        print(f"Train set: {len(X_train):,} samples (bars 0-{split_idx:,})")
        print(f"Test set: {len(X_test):,} samples (bars {split_idx:,}-{len(X):,})")
    else:
        # Random split (WARNING: will overfit for time-series data!)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        print(f"\nUsing RANDOM split (WARNING: May leak future information!)")
        print(f"Train set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")

    # Train XGBoost
    print("\nTraining XGBoost model...")
    print(f"Building {args.n_estimators} trees (boosting rounds)")
    print(f"Max tree depth: {args.max_depth}")
    if args.lr_decay:
        print(f"Note: Learning rate decay not supported in this XGBoost version")
        print(f"  Workaround: Use more trees with lower learning rate")
        print(f"  Recommended: --n-estimators {args.n_estimators * 2} --learning-rate {args.lr_end}")
    if args.early_stopping_rounds:
        print(f"Early stopping: {args.early_stopping_rounds} rounds")

    model = xgb.XGBClassifier(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=args.early_stopping_rounds
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True
    )

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\n" + "="*60)
    print("TRAINING SET PERFORMANCE")
    print("="*60)
    print(classification_report(y_train, y_pred_train, target_names=['Stop Hit', 'Target Hit']))
    print(f"ROC-AUC: {roc_auc_score(y_train, y_pred_proba_train):.4f}")

    print("\n" + "="*60)
    print("TEST SET PERFORMANCE")
    print("="*60)
    print(classification_report(y_test, y_pred_test, target_names=['Stop Hit', 'Target Hit']))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_test):.4f}")

    # Save model
    model_path = os.path.join(args.model_dir, "xgb_1_2_rr.json")
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")

    # Also save with joblib for easier loading
    joblib_path = os.path.join(args.model_dir, "xgb_1_2_rr.pkl")
    joblib.dump(model, joblib_path)
    print(f"Model (joblib) saved to: {joblib_path}")

    # Save feature names
    feature_names_path = os.path.join(args.model_dir, "feature_names.txt")
    with open(feature_names_path, 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"Feature names saved to: {feature_names_path}")

    # Plot feature importance
    plot_feature_importance(
        model,
        feature_cols,
        os.path.join(args.model_dir, "feature_importance.png")
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        y_test,
        y_pred_test,
        os.path.join(args.model_dir, "confusion_matrix.png")
    )

    # Plot precision-recall curve
    plot_precision_recall_curve(
        y_test,
        y_pred_proba_test,
        os.path.join(args.model_dir, "precision_recall_curve.png")
    )

    # Find optimal threshold based on precision-recall tradeoff
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_test)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    print(f"\n Optimal threshold (max F1): {optimal_threshold:.4f}")
    print(f"  Precision: {precision[optimal_idx]:.4f}")
    print(f"  Recall: {recall[optimal_idx]:.4f}")
    print(f"  F1 Score: {f1_scores[optimal_idx]:.4f}")

    # Save threshold
    threshold_path = os.path.join(args.model_dir, "optimal_threshold.txt")
    with open(threshold_path, 'w') as f:
        f.write(f"{optimal_threshold}\n")
    print(f"Optimal threshold saved to: {threshold_path}")


if __name__ == "__main__":
    main()
