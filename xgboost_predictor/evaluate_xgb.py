#!/usr/bin/env python3
"""
Evaluate XGBoost model and simulate trading with predictions.
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime


def simulate_trading(df: pd.DataFrame, predictions: np.ndarray, probabilities: np.ndarray,
                     threshold: float = 0.5, transaction_cost_bps: float = 0.5,
                     contract_multiplier: float = 5.0) -> dict:
    """Simulate trading based on XGBoost predictions.

    Args:
        df: DataFrame with labeled data
        predictions: Binary predictions (1 = take trade, 0 = skip)
        probabilities: Prediction probabilities
        threshold: Probability threshold for taking trades
        transaction_cost_bps: Transaction cost in basis points
        contract_multiplier: Contract multiplier

    Returns:
        Dictionary with trading statistics
    """
    # Filter for high-confidence predictions above threshold
    high_conf_mask = probabilities >= threshold
    trades_taken = high_conf_mask.sum()

    if trades_taken == 0:
        print("No trades taken with this threshold!")
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl_per_trade': 0,
        }

    # Get actual outcomes for trades we would have taken
    actual_outcomes = df['target_hit'].values[high_conf_mask]
    entry_prices = df['entry_price'].values[high_conf_mask]

    # Calculate PnL for each trade
    pnls = []
    for i, (outcome, entry) in enumerate(zip(actual_outcomes, entry_prices)):
        cost = entry * contract_multiplier
        transaction_cost = cost * (transaction_cost_bps / 10000)

        if outcome == 1:  # Target hit
            # Risk was $5, reward is $10 (1:2 RR)
            # But we need to calculate based on actual price movement
            risk_dollars = (entry - df.iloc[high_conf_mask.nonzero()[0][i]]['stop_price']) * contract_multiplier
            reward_dollars = risk_dollars * 2
            pnl = reward_dollars - 2 * transaction_cost
        else:  # Stop hit
            risk_dollars = (entry - df.iloc[high_conf_mask.nonzero()[0][i]]['stop_price']) * contract_multiplier
            pnl = -risk_dollars - 2 * transaction_cost

        pnls.append(pnl)

    pnls = np.array(pnls)
    wins = (pnls > 0).sum()
    losses = (pnls <= 0).sum()

    return {
        'total_trades': trades_taken,
        'wins': wins,
        'losses': losses,
        'win_rate': wins / trades_taken * 100 if trades_taken > 0 else 0,
        'total_pnl': pnls.sum(),
        'avg_pnl_per_trade': pnls.mean(),
        'best_trade': pnls.max(),
        'worst_trade': pnls.min(),
        'pnls': pnls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="xgboost_predictor/labeled_data.csv",
                       help="Labeled data CSV")
    parser.add_argument("--model", default="xgboost_predictor/models/xgb_1_2_rr.pkl",
                       help="Trained XGBoost model")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Prediction threshold (default: load from optimal_threshold.txt)")
    parser.add_argument("--transaction-cost", type=float, default=0.5,
                       help="Transaction cost in bps (default: 0.5)")
    parser.add_argument("--output-dir", default="xgboost_predictor/eval_results",
                       help="Directory to save results")
    parser.add_argument("--test-only", action="store_true",
                       help="Evaluate only on test set (last 20% of data)")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size for test-only mode (default: 0.2)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load threshold if not provided
    if args.threshold is None:
        threshold_path = os.path.join(os.path.dirname(args.model), "optimal_threshold.txt")
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                args.threshold = float(f.read().strip())
            print(f"Loaded optimal threshold: {args.threshold:.4f}")
        else:
            args.threshold = 0.5
            print(f"Using default threshold: {args.threshold}")

    # Load model
    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Total samples: {len(df):,}")

    # Filter for samples with outcomes
    df_with_outcome = df[df['bars_to_outcome'] > 0].copy()
    print(f"Samples with outcome: {len(df_with_outcome):,}")

    # If test-only, use only the last portion of data
    if args.test_only:
        split_idx = int(len(df_with_outcome) * (1 - args.test_size))
        df_with_outcome = df_with_outcome.iloc[split_idx:].copy()
        print(f"\n⚠️  TEST-ONLY MODE: Using only last {args.test_size*100:.0f}% of data")
        print(f"Evaluating on {len(df_with_outcome):,} samples (bars {split_idx:,} onwards)")
        print(f"This represents UNSEEN data if model was trained with --time-series-split")
    else:
        print(f"\n⚠️  WARNING: Evaluating on FULL dataset (includes training data)")
        print(f"Results will be overly optimistic! Use --test-only for realistic performance.")

    # Prepare features
    label_cols = ['target_hit', 'stop_hit', 'bars_to_outcome', 'entry_price', 'stop_price', 'target_price']
    feature_cols = [col for col in df_with_outcome.columns if col not in label_cols]
    X = df_with_outcome[feature_cols].values

    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Basic prediction stats
    print("\n" + "="*60)
    print("PREDICTION STATISTICS")
    print("="*60)
    print(f"Total predictions: {len(predictions):,}")
    print(f"Predicted positive (model says target will hit): {predictions.sum():,} ({predictions.sum()/len(predictions)*100:.1f}%)")
    print(f"Predicted negative (model says stop will hit): {len(predictions) - predictions.sum():,} ({(len(predictions) - predictions.sum())/len(predictions)*100:.1f}%)")
    print(f"Actual positive (target hit): {df_with_outcome['target_hit'].sum():,} ({df_with_outcome['target_hit'].sum()/len(df_with_outcome)*100:.1f}%)")

    # Simulate trading with different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, args.threshold]
    thresholds = sorted(set(thresholds))  # Remove duplicates and sort

    print("\n" + "="*60)
    print("TRADING SIMULATION (Different Thresholds)")
    print("="*60)

    results = []
    for thresh in thresholds:
        stats = simulate_trading(
            df_with_outcome,
            predictions,
            probabilities,
            threshold=thresh,
            transaction_cost_bps=args.transaction_cost
        )
        results.append({
            'threshold': thresh,
            **stats
        })

        print(f"\nThreshold: {thresh:.2f}")
        print(f"  Trades taken: {stats['total_trades']:,}")
        print(f"  Wins: {stats['wins']} | Losses: {stats['losses']}")
        print(f"  Win rate: {stats['win_rate']:.1f}%")
        print(f"  Total PnL: ${stats['total_pnl']:,.2f}")
        print(f"  Avg PnL per trade: ${stats['avg_pnl_per_trade']:.2f}")
        if stats['total_trades'] > 0:
            print(f"  Best trade: ${stats['best_trade']:.2f}")
            print(f"  Worst trade: ${stats['worst_trade']:.2f}")

    # Save results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'pnls'} for r in results])
    results_path = os.path.join(args.output_dir, f"threshold_comparison_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nThreshold comparison saved to: {results_path}")

    # Save detailed predictions
    predictions_df = df_with_outcome.copy()
    predictions_df['predicted_target_hit'] = predictions
    predictions_df['prediction_probability'] = probabilities
    predictions_path = os.path.join(args.output_dir, f"predictions_{timestamp}.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Detailed predictions saved to: {predictions_path}")

    # Print recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    best_result = max(results, key=lambda x: x['total_pnl'])
    print(f"Best performing threshold: {best_result['threshold']:.2f}")
    print(f"  Total PnL: ${best_result['total_pnl']:,.2f}")
    print(f"  Win rate: {best_result['win_rate']:.1f}%")
    print(f"  Trades: {best_result['total_trades']:,}")


if __name__ == "__main__":
    main()
