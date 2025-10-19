#!/usr/bin/env python3
"""
Visualize technical indicators and price data used in training.
Shows the exact features the model sees.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import compute_indicators
from datetime import datetime
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="datasets/mes_finrl_ready_front.csv")
    parser.add_argument("--train-slice", type=int, default=5000, help="Same as training")
    parser.add_argument("--start", type=int, default=0, help="Start bar within slice")
    parser.add_argument("--bars", type=int, default=500, help="Number of bars to plot")
    parser.add_argument("--save", action="store_true", help="Save plot instead of showing")
    args = parser.parse_args()
    
    # Load data (same as training)
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    
    if args.train_slice and args.train_slice < len(df):
        df = df.tail(args.train_slice).reset_index(drop=True)
        print(f"Using last {args.train_slice} bars (same as training)")
    
    # Compute indicators (same as training)
    print("Computing indicators...")
    tech_df = compute_indicators(df)
    
    # Merge with price data
    df['rsi_14'] = tech_df['rsi_14']
    df['sma_7'] = tech_df['sma_7']
    df['sma_21'] = tech_df['sma_21']
    df['sin_tod'] = tech_df['sin_tod']
    df['cos_tod'] = tech_df['cos_tod']
    df['sin_dow'] = tech_df['sin_dow']
    df['cos_dow'] = tech_df['cos_dow']
    df['is_rth'] = tech_df['is_rth']
    df['frac_session'] = tech_df['frac_session']
    
    # Select plotting range
    end = min(args.start + args.bars, len(df))
    plot_df = df.iloc[args.start:end].copy()
    
    # Parse timestamps
    plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Layout: 5 rows
    # Row 1: Price + SMAs
    # Row 2: RSI
    # Row 3: Time features (sin/cos)
    # Row 4: RTH and session fraction
    # Row 5: Volume
    
    # 1. Price and SMAs
    ax1 = plt.subplot(5, 1, 1)
    ax1.plot(plot_df['timestamp'], plot_df['close'], 'b-', linewidth=1.5, label='Close Price')
    ax1.plot(plot_df['timestamp'], plot_df['sma_7'], 'g--', linewidth=1, label='SMA 7', alpha=0.8)
    ax1.plot(plot_df['timestamp'], plot_df['sma_21'], 'r--', linewidth=1, label='SMA 21', alpha=0.8)
    ax1.fill_between(plot_df['timestamp'], plot_df['sma_7'], plot_df['sma_21'], 
                     where=(plot_df['sma_7'] > plot_df['sma_21']), 
                     interpolate=True, alpha=0.2, color='green', label='SMA7 > SMA21')
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'MES Price and Moving Averages ({args.bars} bars from {args.start})')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. RSI
    ax2 = plt.subplot(5, 1, 2, sharex=ax1)
    ax2.plot(plot_df['timestamp'], plot_df['rsi_14'], 'purple', linewidth=1.5, label='RSI(14)')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
    ax2.fill_between(plot_df['timestamp'], 70, plot_df['rsi_14'], 
                     where=(plot_df['rsi_14'] > 70), interpolate=True, alpha=0.3, color='red')
    ax2.fill_between(plot_df['timestamp'], 30, plot_df['rsi_14'], 
                     where=(plot_df['rsi_14'] < 30), interpolate=True, alpha=0.3, color='green')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Time features (cyclical)
    ax3 = plt.subplot(5, 1, 3, sharex=ax1)
    ax3.plot(plot_df['timestamp'], plot_df['sin_tod'], 'orange', linewidth=1, label='sin(time of day)')
    ax3.plot(plot_df['timestamp'], plot_df['cos_tod'], 'brown', linewidth=1, label='cos(time of day)')
    ax3.plot(plot_df['timestamp'], plot_df['sin_dow'], 'cyan', linewidth=1, label='sin(day of week)', alpha=0.7)
    ax3.plot(plot_df['timestamp'], plot_df['cos_dow'], 'magenta', linewidth=1, label='cos(day of week)', alpha=0.7)
    ax3.set_ylabel('Cyclical Features')
    ax3.set_ylim(-1.1, 1.1)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.legend(loc='upper left', ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # 4. RTH flag and session fraction
    ax4 = plt.subplot(5, 1, 4, sharex=ax1)
    # RTH as shaded regions
    rth_starts = []
    rth_ends = []
    in_rth = False
    for i in range(len(plot_df)):
        if plot_df['is_rth'].iloc[i] and not in_rth:
            rth_starts.append(plot_df['timestamp'].iloc[i])
            in_rth = True
        elif not plot_df['is_rth'].iloc[i] and in_rth:
            rth_ends.append(plot_df['timestamp'].iloc[i])
            in_rth = False
    if in_rth and len(rth_ends) < len(rth_starts):
        rth_ends.append(plot_df['timestamp'].iloc[-1])
    
    for start, end in zip(rth_starts, rth_ends):
        ax4.axvspan(start, end, alpha=0.3, color='yellow', label='RTH' if start == rth_starts[0] else '')
    
    ax4.plot(plot_df['timestamp'], plot_df['frac_session'], 'darkgreen', linewidth=1.5, label='Session Fraction')
    ax4.set_ylabel('Session Progress')
    ax4.set_ylim(-0.1, 1.1)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Volume
    ax5 = plt.subplot(5, 1, 5, sharex=ax1)
    ax5.bar(plot_df['timestamp'], plot_df['volume'], width=0.0007, alpha=0.6, color='gray')
    ax5.set_ylabel('Volume')
    ax5.set_xlabel('Time')
    ax5.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Summary stats
    print("\nIndicator Summary:")
    print(f"Price range: ${plot_df['close'].min():.2f} - ${plot_df['close'].max():.2f}")
    print(f"RSI range: {plot_df['rsi_14'].min():.1f} - {plot_df['rsi_14'].max():.1f}")
    print(f"RSI mean: {plot_df['rsi_14'].mean():.1f}")
    print(f"RTH percentage: {plot_df['is_rth'].mean()*100:.1f}%")
    print(f"SMA7 > SMA21: {(plot_df['sma_7'] > plot_df['sma_21']).mean()*100:.1f}% of time")
    
    # Model's observation at a sample point
    mid_point = len(plot_df) // 2
    print(f"\nSample observation at bar {args.start + mid_point}:")
    print(f"  close: {plot_df['close'].iloc[mid_point]:.2f}")
    print(f"  rsi_14: {plot_df['rsi_14'].iloc[mid_point]:.1f}")
    print(f"  sma_7: {plot_df['sma_7'].iloc[mid_point]:.2f}")
    print(f"  sma_21: {plot_df['sma_21'].iloc[mid_point]:.2f}")
    print(f"  sin_tod: {plot_df['sin_tod'].iloc[mid_point]:.3f}")
    print(f"  cos_tod: {plot_df['cos_tod'].iloc[mid_point]:.3f}")
    print(f"  is_rth: {plot_df['is_rth'].iloc[mid_point]}")
    print(f"  frac_session: {plot_df['frac_session'].iloc[mid_point]:.3f}")
    
    if args.save:
        filename = f"indicators_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
