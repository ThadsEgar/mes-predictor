#!/usr/bin/env python3
"""
Hyperparameter sweep for XGBoost model.
Automatically finds the best parameters and trains the final model.
"""

import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import os
from datetime import datetime
import joblib
from itertools import product
import json


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'accuracy': (y_pred == y_test).mean()
    }


def grid_search(X_train, y_train, X_test, y_test, param_grid, scale_pos_weight, verbose=True):
    """Perform grid search over parameter space."""
    results = []

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    total = len(combinations)
    print(f"\nTesting {total} parameter combinations...")
    print("="*80)

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))

        if verbose:
            print(f"\n[{i}/{total}] Testing: {params}")

        # Train model
        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train, verbose=False)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        result = {**params, **metrics}
        results.append(result)

        if verbose:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"Precision: {metrics['precision']:.4f} | "
                  f"Recall: {metrics['recall']:.4f}")

    return results


def random_search(X_train, y_train, X_test, y_test, param_distributions,
                  scale_pos_weight, n_iter=20, verbose=True):
    """Perform random search over parameter space."""
    results = []

    print(f"\nTesting {n_iter} random parameter combinations...")
    print("="*80)

    for i in range(n_iter):
        # Sample random parameters
        params = {}
        for param_name, param_values in param_distributions.items():
            if isinstance(param_values, list):
                params[param_name] = np.random.choice(param_values)
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # Assume (min, max) for continuous values
                if isinstance(param_values[0], int):
                    params[param_name] = np.random.randint(param_values[0], param_values[1] + 1)
                else:
                    params[param_name] = np.random.uniform(param_values[0], param_values[1])

        if verbose:
            print(f"\n[{i+1}/{n_iter}] Testing: {params}")

        # Train model
        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train, verbose=False)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        result = {**params, **metrics}
        results.append(result)

        if verbose:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"Precision: {metrics['precision']:.4f} | "
                  f"Recall: {metrics['recall']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="xgboost_predictor/labeled_data.csv",
                       help="Labeled data CSV")
    parser.add_argument("--output-dir", default="xgboost_predictor/sweep_results",
                       help="Directory to save sweep results")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size (default: 0.2)")
    parser.add_argument("--search-type", choices=['grid', 'random'], default='random',
                       help="Search type: grid or random (default: random)")
    parser.add_argument("--n-iter", type=int, default=50,
                       help="Number of iterations for random search (default: 50)")
    parser.add_argument("--metric", choices=['roc_auc', 'f1', 'precision', 'recall'],
                       default='roc_auc',
                       help="Metric to optimize (default: roc_auc)")
    parser.add_argument("--train-best", action="store_true",
                       help="Automatically train final model with best parameters")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Total samples: {len(df):,}")

    # Filter for samples with outcomes
    df_with_outcome = df[df['bars_to_outcome'] > 0].copy()
    print(f"Samples with outcome: {len(df_with_outcome):,}")

    # Prepare features and labels
    label_cols = ['target_hit', 'stop_hit', 'bars_to_outcome', 'entry_price', 'stop_price', 'target_price']
    feature_cols = [col for col in df_with_outcome.columns if col not in label_cols]

    X = df_with_outcome[feature_cols].values
    y = df_with_outcome['target_hit'].values

    # Time-series split
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTime-series split:")
    print(f"  Train: {len(X_train):,} samples (bars 0-{split_idx:,})")
    print(f"  Test: {len(X_test):,} samples (bars {split_idx:,}-{len(X):,})")

    # Calculate scale_pos_weight
    neg_count = len(y_train) - y_train.sum()
    pos_count = y_train.sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")

    # Define parameter space
    if args.search_type == 'grid':
        # Grid search: test all combinations (smaller grid)
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.03, 0.05, 0.07],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
        }

        results = grid_search(X_train, y_train, X_test, y_test, param_grid, scale_pos_weight)

    else:  # random search
        # Random search: sample from distributions
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500, 600, 800],
            'max_depth': [4, 6, 8, 10, 12, 15],
            'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7, 10],
        }

        results = random_search(X_train, y_train, X_test, y_test, param_distributions,
                               scale_pos_weight, n_iter=args.n_iter)

    # Convert to DataFrame and sort by metric
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=args.metric, ascending=False)

    # Save results
    results_path = os.path.join(args.output_dir, f"sweep_{args.search_type}_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_path}")

    # Print top 10 results
    print(f"\n{'='*80}")
    print(f"TOP 10 RESULTS (sorted by {args.metric}):")
    print("="*80)
    print(results_df.head(10).to_string(index=False))

    # Best parameters
    best_params = results_df.iloc[0].to_dict()
    param_keys = ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
                  'colsample_bytree', 'min_child_weight']
    best_params_only = {k: best_params[k] for k in param_keys if k in best_params}

    # Ensure integer parameters are integers (fix for JSON/numpy float conversion)
    int_params = ['n_estimators', 'max_depth', 'min_child_weight']
    for param in int_params:
        if param in best_params_only:
            best_params_only[param] = int(best_params_only[param])

    print(f"\n{'='*80}")
    print("BEST PARAMETERS:")
    print("="*80)
    for k, v in best_params_only.items():
        print(f"  {k}: {v}")

    print(f"\nBest {args.metric}: {best_params[args.metric]:.4f}")
    print(f"  ROC-AUC: {best_params['roc_auc']:.4f}")
    print(f"  F1 Score: {best_params['f1']:.4f}")
    print(f"  Precision: {best_params['precision']:.4f}")
    print(f"  Recall: {best_params['recall']:.4f}")

    # Save best parameters
    best_params_path = os.path.join(args.output_dir, f"best_params_{timestamp}.json")
    with open(best_params_path, 'w') as f:
        json.dump(best_params_only, f, indent=2)
    print(f"\nBest parameters saved to: {best_params_path}")

    # Train final model with best parameters
    if args.train_best:
        print(f"\n{'='*80}")
        print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
        print("="*80)

        final_model = xgb.XGBClassifier(
            **best_params_only,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )

        final_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=True
        )

        # Save model
        model_dir = "xgboost_predictor/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"xgb_1_2_rr_best.pkl")
        joblib.dump(final_model, model_path)
        print(f"\nBest model saved to: {model_path}")

        # Save feature names
        feature_names_path = os.path.join(model_dir, "feature_names.txt")
        with open(feature_names_path, 'w') as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        print(f"Feature names saved to: {feature_names_path}")

    # Print command to train manually
    print(f"\n{'='*80}")
    print("TO TRAIN WITH BEST PARAMETERS MANUALLY:")
    print("="*80)
    print("python xgboost_predictor/train_xgb.py \\")
    print("  --time-series-split \\")
    for k, v in best_params_only.items():
        print(f"  --{k.replace('_', '-')} {v} \\")
    print("  --early-stopping-rounds 50")


if __name__ == "__main__":
    main()
