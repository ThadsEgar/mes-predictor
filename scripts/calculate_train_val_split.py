#!/usr/bin/env python3
"""
Calculate train/validation split leaving last 3 months for validation.
Assumes minute-level data with standard trading hours.
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="datasets/mes_finrl_ready_front.csv")
    parser.add_argument("--val-months", type=int, default=3,
                       help="Number of months to reserve for validation (default: 3)")
    parser.add_argument("--trading-days-per-month", type=int, default=21,
                       help="Trading days per month (default: 21)")
    parser.add_argument("--minutes-per-day", type=int, default=390,
                       help="Trading minutes per day (default: 390 = 6.5 hours)")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    total_bars = len(df)

    print(f"\n{'='*70}")
    print("TRAIN/VALIDATION SPLIT CALCULATION")
    print('='*70)
    print(f"Total bars in dataset: {total_bars:,}")

    # Calculate validation set size
    val_trading_days = args.val_months * args.trading_days_per_month
    val_bars = val_trading_days * args.minutes_per_day

    print(f"\nValidation period: {args.val_months} months")
    print(f"  Trading days: {val_trading_days}")
    print(f"  Minutes per day: {args.minutes_per_day}")
    print(f"  Total validation bars: {val_bars:,}")

    # Calculate train set size
    train_bars = total_bars - val_bars

    if train_bars <= 0:
        print(f"\n❌ ERROR: Not enough data!")
        print(f"   Need at least {val_bars:,} bars, but only have {total_bars:,}")
        return

    print(f"\nTrain set: {train_bars:,} bars ({train_bars/total_bars*100:.1f}%)")
    print(f"Validation set: {val_bars:,} bars ({val_bars/total_bars*100:.1f}%)")

    # Check if we have a date column to show actual dates
    if 'date' in df.columns or 'timestamp' in df.columns:
        date_col = 'date' if 'date' in df.columns else 'timestamp'
        print(f"\n{'='*70}")
        print("DATE RANGES")
        print('='*70)

        # Full dataset
        print(f"Full dataset:")
        print(f"  Start: {df[date_col].iloc[0]}")
        print(f"  End:   {df[date_col].iloc[-1]}")

        # Training set
        print(f"\nTraining set (bars 0 to {train_bars:,}):")
        print(f"  Start: {df[date_col].iloc[0]}")
        print(f"  End:   {df[date_col].iloc[train_bars-1]}")

        # Validation set
        print(f"\nValidation set (bars {train_bars:,} to {total_bars:,}):")
        print(f"  Start: {df[date_col].iloc[train_bars]}")
        print(f"  End:   {df[date_col].iloc[-1]}")

    print(f"\n{'='*70}")
    print("TRAINING COMMANDS")
    print('='*70)

    print("\n1. Train model (on all data EXCEPT last 3 months):")
    print(f"""
python scripts/train_dense.py \\
  --name rl_model_v13_train_val_split \\
  --cost-start 0.5 \\
  --cost-end 0.5 \\
  --timesteps 3_000_000_000 \\
  --train-slice {train_bars} \\
  --n-envs 32 \\
  --learning-rate 5e-5 \\
  --lr-decay \\
  --lr-end 1e-6 \\
  --ent-coef 0.03
""")

    print("\n2. Evaluate on validation set (last 3 months - UNSEEN):")
    print(f"""
python scripts/eval_test_set.py \\
  --model models/rl_model_v13_train_val_split \\
  --total-bars {total_bars} \\
  --train-size {train_bars/total_bars:.4f} \\
  --transaction-cost 0.5
""")

    print("\n3. Or use the quick validation script:")
    print(f"""
python scripts/validate_model.py \\
  --model models/rl_model_v13_train_val_split \\
  --train-bars {train_bars}
""")

    # Save to file for easy reference
    with open('models/train_val_split_config.txt', 'w') as f:
        f.write(f"Total bars: {total_bars}\n")
        f.write(f"Train bars: {train_bars}\n")
        f.write(f"Validation bars: {val_bars}\n")
        f.write(f"Validation months: {args.val_months}\n")
        f.write(f"Train fraction: {train_bars/total_bars:.4f}\n")

    print(f"\n✓ Split configuration saved to: models/train_val_split_config.txt")
    print('='*70)


if __name__ == "__main__":
    main()
