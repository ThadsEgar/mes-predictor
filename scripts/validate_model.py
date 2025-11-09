#!/usr/bin/env python3
"""
Quick validation script - evaluates model on held-out validation set.
"""

import argparse
import subprocess
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model to validate")
    parser.add_argument("--train-bars", type=int, required=True,
                       help="Number of bars used for training (rest is validation)")
    parser.add_argument("--csv", default="datasets/mes_finrl_ready_front.csv")
    parser.add_argument("--transaction-cost", type=float, default=0.5)
    args = parser.parse_args()

    # Check if split config exists
    config_file = "models/train_val_split_config.txt"
    if os.path.exists(config_file):
        print("Using split configuration from:")
        print(f"  {config_file}\n")
        with open(config_file, 'r') as f:
            print(f.read())

    # Load dataset to get total bars
    import pandas as pd
    df = pd.read_csv(args.csv)
    total_bars = len(df)
    val_bars = total_bars - args.train_bars
    train_fraction = args.train_bars / total_bars

    print(f"\n{'='*70}")
    print("VALIDATION SETUP")
    print('='*70)
    print(f"Model: {args.model}")
    print(f"Total bars: {total_bars:,}")
    print(f"Train bars: {args.train_bars:,} ({train_fraction*100:.1f}%)")
    print(f"Validation bars: {val_bars:,} ({(1-train_fraction)*100:.1f}%)")
    print(f"Transaction cost: {args.transaction_cost} bps")
    print('='*70)

    # Run evaluation on test set
    cmd = [
        "python", "scripts/eval_test_set.py",
        "--model", args.model,
        "--total-bars", str(total_bars),
        "--train-size", f"{train_fraction:.6f}",
        "--transaction-cost", str(args.transaction_cost),
        "--csv", args.csv
    ]

    print("\nRunning validation...")
    print(f"Command: {' '.join(cmd)}\n")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
