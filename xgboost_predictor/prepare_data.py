#!/usr/bin/env python3
"""
Prepare labeled data for XGBoost to predict 1:2 risk-reward trades.

For each timestep, we label whether entering a long position would achieve
a 1:2 risk-reward ratio (target hit before stop loss).
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils import compute_indicators


def label_1_2_rr_trades(df: pd.DataFrame, risk_dollars: float = 5.0, contract_multiplier: float = 5.0,
                        tick_size: float = 0.25, lookahead_bars: int = 240) -> pd.DataFrame:
    """Label each timestep with whether a 1:2 RR trade is achievable.

    Args:
        df: DataFrame with OHLC data
        risk_dollars: Risk amount in dollars (e.g., $5)
        contract_multiplier: Contract multiplier (e.g., 5.0 for MES)
        tick_size: Tick size (e.g., 0.25 for MES)
        lookahead_bars: Maximum bars to look ahead for target/stop (default: 240 = 4 hours)

    Returns:
        DataFrame with additional columns:
        - target_hit: 1 if target was hit before stop, 0 otherwise
        - stop_hit: 1 if stop was hit first, 0 otherwise
        - bars_to_outcome: Number of bars until outcome (target or stop hit)
        - entry_price: Entry price at this timestep
        - stop_price: Stop loss price
        - target_price: Target price (2x risk)
    """
    print(f"Labeling data with 1:2 RR trades...")
    print(f"  Risk: ${risk_dollars}")
    print(f"  Contract multiplier: {contract_multiplier}")
    print(f"  Tick size: ${tick_size}")
    print(f"  Lookahead window: {lookahead_bars} bars")

    # Convert risk in dollars to price points
    risk_price = risk_dollars / contract_multiplier
    target_price_distance = risk_price * 2  # 1:2 RR means 2x the risk

    print(f"  Risk in price points: ${risk_price:.2f}")
    print(f"  Target distance: ${target_price_distance:.2f}")

    # Initialize result columns
    target_hit = np.zeros(len(df), dtype=int)
    stop_hit = np.zeros(len(df), dtype=int)
    bars_to_outcome = np.full(len(df), -1, dtype=int)
    entry_prices = df['close'].values.copy()
    stop_prices = entry_prices - risk_price
    target_prices = entry_prices + target_price_distance

    # For each bar, look ahead to see if target or stop is hit first
    for i in range(len(df) - 1):
        entry = df['close'].iloc[i]
        stop = entry - risk_price
        target = entry + target_price_distance

        # Look ahead up to lookahead_bars
        end_idx = min(i + 1 + lookahead_bars, len(df))
        future_highs = df['high'].iloc[i+1:end_idx].values
        future_lows = df['low'].iloc[i+1:end_idx].values

        # Check each future bar
        for j, (high, low) in enumerate(zip(future_highs, future_lows)):
            # Check if stop is hit (assume stop is checked first in same bar)
            if low <= stop:
                stop_hit[i] = 1
                target_hit[i] = 0
                bars_to_outcome[i] = j + 1
                break
            # Check if target is hit
            elif high >= target:
                target_hit[i] = 1
                stop_hit[i] = 0
                bars_to_outcome[i] = j + 1
                break

    # Add labels to dataframe
    df['target_hit'] = target_hit
    df['stop_hit'] = stop_hit
    df['bars_to_outcome'] = bars_to_outcome
    df['entry_price'] = entry_prices
    df['stop_price'] = stop_prices
    df['target_price'] = target_prices

    # Stats
    total_labeled = (target_hit + stop_hit).sum()
    target_hit_count = target_hit.sum()
    stop_hit_count = stop_hit.sum()
    no_outcome_count = len(df) - total_labeled

    print(f"\nLabeling statistics:")
    print(f"  Total bars: {len(df):,}")
    print(f"  Target hit first: {target_hit_count:,} ({target_hit_count/len(df)*100:.1f}%)")
    print(f"  Stop hit first: {stop_hit_count:,} ({stop_hit_count/len(df)*100:.1f}%)")
    print(f"  No outcome: {no_outcome_count:,} ({no_outcome_count/len(df)*100:.1f}%)")
    if total_labeled > 0:
        print(f"  Win rate (of decided): {target_hit_count/total_labeled*100:.1f}%")
    print(f"  Avg bars to outcome: {df[df['bars_to_outcome'] > 0]['bars_to_outcome'].mean():.1f}")

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature columns for XGBoost.

    Args:
        df: DataFrame with OHLC and technical indicators

    Returns:
        DataFrame with feature columns ready for XGBoost
    """
    # Get technical indicators
    tech = compute_indicators(df)

    # Combine price data and technical indicators
    features = pd.DataFrame({
        # Price features
        'close': df['close'],
        'high': df['high'],
        'low': df['low'],
        'open': df['open'],

        # Technical indicators
        'rsi_14': tech['rsi_14'],
        'sma_7': tech['sma_7'],
        'sma_21': tech['sma_21'],
        'atr_14': tech['atr_14'],

        # Price relationships
        'close_to_sma7': df['close'] - tech['sma_7'],
        'close_to_sma21': df['close'] - tech['sma_21'],
        'sma7_to_sma21': tech['sma_7'] - tech['sma_21'],

        # Volatility features
        'high_low_range': df['high'] - df['low'],
        'close_to_open': df['close'] - df['open'],

        # Momentum features
        'price_change_1': df['close'].diff(1),
        'price_change_5': df['close'].diff(5),
        'price_change_20': df['close'].diff(20),
    })

    # Fill NaN values
    features = features.bfill().ffill().fillna(0.0)

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="datasets/mes_finrl_ready_front.csv",
                       help="Input CSV with OHLC data")
    parser.add_argument("--output", default="xgboost_predictor/labeled_data.csv",
                       help="Output CSV with labeled data")
    parser.add_argument("--risk-dollars", type=float, default=5.0,
                       help="Risk amount in dollars (default: 5.0)")
    parser.add_argument("--contract-multiplier", type=float, default=5.0,
                       help="Contract multiplier (default: 5.0 for MES)")
    parser.add_argument("--tick-size", type=float, default=0.25,
                       help="Tick size (default: 0.25 for MES)")
    parser.add_argument("--lookahead-bars", type=int, default=240,
                       help="Maximum bars to look ahead (default: 240 = 4 hours)")
    parser.add_argument("--train-slice", type=int, default=None,
                       help="Use only last N bars (optional)")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)

    # Use slice if specified
    if args.train_slice and args.train_slice < len(df):
        print(f"Using last {args.train_slice} bars")
        df = df.tail(args.train_slice).reset_index(drop=True)

    print(f"Total bars: {len(df):,}")

    # Check required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Label the data
    df_labeled = label_1_2_rr_trades(
        df,
        risk_dollars=args.risk_dollars,
        contract_multiplier=args.contract_multiplier,
        tick_size=args.tick_size,
        lookahead_bars=args.lookahead_bars
    )

    # Prepare features
    print("\nPreparing features...")
    features = prepare_features(df_labeled)

    # Combine features and labels
    output_df = pd.concat([features, df_labeled[['target_hit', 'stop_hit', 'bars_to_outcome',
                                                   'entry_price', 'stop_price', 'target_price']]], axis=1)

    # Save
    print(f"\nSaving to {args.output}...")
    output_df.to_csv(args.output, index=False)
    print(f"Saved {len(output_df):,} rows with {len(features.columns)} features")

    print("\nFeature columns:")
    for col in features.columns:
        print(f"  - {col}")

    print("\nLabel columns:")
    print("  - target_hit (1 if 1:2 RR achieved, 0 otherwise)")
    print("  - stop_hit (1 if stop hit first)")
    print("  - bars_to_outcome (bars until outcome)")


if __name__ == "__main__":
    main()
