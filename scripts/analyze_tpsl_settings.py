#!/usr/bin/env python3
"""
Analyze TP/SL settings for scalping on 1-minute data.

Tests different stop_ticks and RR ratios by simulating random entries
and tracking how often TP vs SL is hit, and how long trades take.

Usage:
    python scripts/analyze_tpsl_settings.py --csv datasets/mes_finrl_ready_front.csv
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def simulate_trade(
    prices: np.ndarray,
    entry_idx: int,
    entry_price: float,
    stop_ticks: int,
    rr: float,
    tick_size: float = 0.25,
    max_bars: int = 500,
) -> Tuple[str, int, float]:
    """
    Simulate a single trade from entry to TP/SL exit.
    
    Returns:
        (outcome, bars_held, pnl_ticks) where outcome is "TP", "SL", or "timeout"
    """
    stop_dist = stop_ticks * tick_size
    target_dist = rr * stop_dist
    
    stop_price = entry_price - stop_dist
    target_price = entry_price + target_dist
    
    for i in range(1, min(max_bars, len(prices) - entry_idx)):
        price = prices[entry_idx + i]
        
        # Check TP/SL (stop-first precedence)
        if price <= stop_price:
            pnl_ticks = -stop_ticks
            return "SL", i, pnl_ticks
        elif price >= target_price:
            pnl_ticks = stop_ticks * rr
            return "TP", i, pnl_ticks
    
    # Timeout - force exit at current price
    if entry_idx + max_bars < len(prices):
        exit_price = prices[entry_idx + max_bars]
        pnl_ticks = (exit_price - entry_price) / tick_size
        return "timeout", max_bars, pnl_ticks
    
    return "timeout", len(prices) - entry_idx - 1, 0.0


def test_config(
    prices: np.ndarray,
    stop_ticks: int,
    rr: float,
    tick_size: float = 0.25,
    n_samples: int = 1000,
    max_bars: int = 500,
    seed: int = 42,
) -> Dict:
    """Test a specific TP/SL configuration with random entries."""
    np.random.seed(seed)
    
    # Sample random entry points (avoid last max_bars)
    valid_entries = len(prices) - max_bars - 1
    entry_indices = np.random.randint(0, valid_entries, size=n_samples)
    
    results = {
        "TP": 0,
        "SL": 0,
        "timeout": 0,
        "bars_to_tp": [],
        "bars_to_sl": [],
        "bars_to_timeout": [],
        "pnl_ticks": [],
    }
    
    for entry_idx in entry_indices:
        entry_price = prices[entry_idx]
        outcome, bars, pnl = simulate_trade(
            prices, entry_idx, entry_price, stop_ticks, rr, tick_size, max_bars
        )
        
        results[outcome] += 1
        results["pnl_ticks"].append(pnl)
        
        if outcome == "TP":
            results["bars_to_tp"].append(bars)
        elif outcome == "SL":
            results["bars_to_sl"].append(bars)
        else:
            results["bars_to_timeout"].append(bars)
    
    return results


def print_results(config: Dict, results: Dict, n_samples: int):
    """Pretty print results for a configuration."""
    tp_count = results["TP"]
    sl_count = results["SL"]
    timeout_count = results["timeout"]
    
    win_rate = tp_count / (tp_count + sl_count) if (tp_count + sl_count) > 0 else 0
    avg_pnl = np.mean(results["pnl_ticks"])
    
    avg_bars_tp = np.mean(results["bars_to_tp"]) if results["bars_to_tp"] else 0
    avg_bars_sl = np.mean(results["bars_to_sl"]) if results["bars_to_sl"] else 0
    
    print(f"\n{'='*70}")
    print(f"Config: stop_ticks={config['stop_ticks']}, RR={config['rr']:.1f}x")
    print(f"  Stop: {config['stop_ticks'] * config['tick_size']:.2f} pts, "
          f"Target: {config['stop_ticks'] * config['rr'] * config['tick_size']:.2f} pts")
    print(f"{'='*70}")
    print(f"Outcomes (n={n_samples}):")
    print(f"  TP:      {tp_count:4d} ({tp_count/n_samples*100:5.1f}%)")
    print(f"  SL:      {sl_count:4d} ({sl_count/n_samples*100:5.1f}%)")
    print(f"  Timeout: {timeout_count:4d} ({timeout_count/n_samples*100:5.1f}%)")
    print(f"\nMetrics:")
    print(f"  Win Rate:           {win_rate*100:5.1f}% (excludes timeouts)")
    print(f"  Avg PnL:            {avg_pnl:+6.2f} ticks")
    print(f"  Avg Bars to TP:     {avg_bars_tp:6.1f} mins")
    print(f"  Avg Bars to SL:     {avg_bars_sl:6.1f} mins")
    
    # Expected value calculation
    breakeven_wr = 1.0 / (1.0 + config['rr'])
    print(f"\n  Break-even WR:      {breakeven_wr*100:5.1f}%")
    if win_rate > breakeven_wr:
        print(f"  ✓ Profitable with random entries!")
    else:
        print(f"  ✗ Unprofitable with random entries (need model edge)")


def main():
    parser = argparse.ArgumentParser(description="Analyze TP/SL settings for scalping")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV with price data (must have 'close' column)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of random entries to test per config (default: 1000)",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=500,
        help="Max bars to hold before timeout (default: 500)",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=0.25,
        help="Tick size for MES (default: 0.25)",
    )
    parser.add_argument(
        "--stop-ticks",
        type=str,
        default="2,4,8,16",
        help="Comma-separated stop_ticks to test (default: 2,4,8,16)",
    )
    parser.add_argument(
        "--rr-ratios",
        type=str,
        default="1.0,1.5,2.0,3.0",
        help="Comma-separated RR ratios to test (default: 1.0,1.5,2.0,3.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    if "close" not in df.columns:
        raise ValueError("CSV must have a 'close' column")
    
    prices = df["close"].values
    print(f"Loaded {len(prices):,} bars of price data")
    
    # Parse configs
    stop_ticks_list = [int(x) for x in args.stop_ticks.split(",")]
    rr_ratios = [float(x) for x in args.rr_ratios.split(",")]
    
    print(f"\nTesting {len(stop_ticks_list)} stop sizes × {len(rr_ratios)} RR ratios")
    print(f"Total configurations: {len(stop_ticks_list) * len(rr_ratios)}")
    
    # Test all configurations
    all_results = []
    
    for stop_ticks in stop_ticks_list:
        for rr in rr_ratios:
            config = {
                "stop_ticks": stop_ticks,
                "rr": rr,
                "tick_size": args.tick_size,
            }
            
            results = test_config(
                prices,
                stop_ticks,
                rr,
                args.tick_size,
                args.n_samples,
                args.max_bars,
                args.seed,
            )
            
            print_results(config, results, args.n_samples)
            
            # Store summary
            tp_count = results["TP"]
            sl_count = results["SL"]
            win_rate = tp_count / (tp_count + sl_count) if (tp_count + sl_count) > 0 else 0
            avg_pnl = np.mean(results["pnl_ticks"])
            avg_bars_tp = np.mean(results["bars_to_tp"]) if results["bars_to_tp"] else 0
            avg_bars_sl = np.mean(results["bars_to_sl"]) if results["bars_to_sl"] else 0
            
            all_results.append({
                "stop_ticks": stop_ticks,
                "stop_pts": stop_ticks * args.tick_size,
                "rr": rr,
                "target_pts": stop_ticks * rr * args.tick_size,
                "win_rate": win_rate,
                "avg_pnl_ticks": avg_pnl,
                "avg_bars_tp": avg_bars_tp,
                "avg_bars_sl": avg_bars_sl,
                "tp_count": tp_count,
                "sl_count": sl_count,
                "timeout_count": results["timeout"],
            })
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Best Configurations")
    print(f"{'='*70}")
    
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values("avg_pnl_ticks", ascending=False)
    
    print("\nTop 5 by Expected PnL (with random entries):")
    print(summary_df.head(5).to_string(index=False))
    
    print("\n\nRecommendations:")
    print("-" * 70)
    
    # Find configs with reasonable trade duration and positive EV
    good_configs = summary_df[
        (summary_df["avg_pnl_ticks"] > 0) &
        (summary_df["avg_bars_tp"] < 200) &
        (summary_df["avg_bars_sl"] < 200)
    ]
    
    if len(good_configs) > 0:
        best = good_configs.iloc[0]
        print(f"✓ Best config for scalping (fast + profitable with random entries):")
        print(f"  --stop-ticks {int(best['stop_ticks'])} --rr {best['rr']:.1f}")
        print(f"  ({best['stop_pts']:.2f} pt stop, {best['target_pts']:.2f} pt target)")
        print(f"  Win rate: {best['win_rate']*100:.1f}%, Avg PnL: {best['avg_pnl_ticks']:+.2f} ticks")
    else:
        print("⚠ No configs are profitable with random entries.")
        print("  This is normal - your model needs to find an edge!")
        
        # Recommend fastest resolution
        fastest = summary_df.loc[summary_df["avg_bars_tp"].idxmin()]
        print(f"\n  Fastest config (for quick feedback during training):")
        print(f"  --stop-ticks {int(fastest['stop_ticks'])} --rr {fastest['rr']:.1f}")
        print(f"  Avg bars to TP: {fastest['avg_bars_tp']:.0f}, to SL: {fastest['avg_bars_sl']:.0f}")
    
    print(f"\n{'='*70}")
    print("Note: Random entries are a baseline. Your model should beat this!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

