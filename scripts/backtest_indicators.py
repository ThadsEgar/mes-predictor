#!/usr/bin/env python3
"""
Backtest simple indicator-based strategies on 1-minute data.

Tests RSI and SMA strategies with the same TP/SL logic as the RL model
to see if these indicators have predictive power for MES scalping.

Usage:
    python scripts/backtest_indicators.py --csv datasets/mes_finrl_ready_front.csv
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI and SMA indicators (matching training setup)."""
    close = df["close"].astype(float)
    
    # RSI(14)
    try:
        import talib  # type: ignore
        rsi = pd.Series(talib.RSI(close.values, timeperiod=14), index=df.index)
    except Exception:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14.0, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14.0, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
    
    # SMAs
    sma7 = close.rolling(window=7, min_periods=1).mean()
    sma21 = close.rolling(window=21, min_periods=1).mean()
    
    indicators = pd.DataFrame({
        "rsi": rsi,
        "sma7": sma7,
        "sma21": sma21,
    }, index=df.index)
    
    return indicators


def simulate_trade(
    prices: np.ndarray,
    entry_idx: int,
    entry_price: float,
    stop_ticks: int,
    rr: float,
    tick_size: float = 0.25,
    max_bars: int = 500,
) -> Tuple[str, int, float, float]:
    """
    Simulate a single trade from entry to TP/SL exit.
    
    Returns:
        (outcome, bars_held, pnl_ticks, exit_price)
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
            return "SL", i, pnl_ticks, price
        elif price >= target_price:
            pnl_ticks = stop_ticks * rr
            return "TP", i, pnl_ticks, price
    
    # Timeout - force exit at current price
    if entry_idx + max_bars < len(prices):
        exit_price = prices[entry_idx + max_bars]
        pnl_ticks = (exit_price - entry_price) / tick_size
        return "timeout", max_bars, pnl_ticks, exit_price
    
    return "timeout", len(prices) - entry_idx - 1, 0.0, entry_price


class Strategy:
    """Base class for indicator strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def should_enter(self, idx: int, df: pd.DataFrame, indicators: pd.DataFrame) -> bool:
        """Returns True if strategy should enter at this bar."""
        raise NotImplementedError


class RSIOversoldStrategy(Strategy):
    """Buy when RSI crosses above oversold threshold."""
    
    def __init__(self, threshold: float = 30.0):
        super().__init__(f"RSI_Oversold_{int(threshold)}")
        self.threshold = threshold
    
    def should_enter(self, idx: int, df: pd.DataFrame, indicators: pd.DataFrame) -> bool:
        if idx < 1:
            return False
        
        rsi_prev = indicators.iloc[idx - 1]["rsi"]
        rsi_now = indicators.iloc[idx]["rsi"]
        
        # RSI crosses above threshold (was below, now above)
        if pd.notna(rsi_prev) and pd.notna(rsi_now):
            return rsi_prev <= self.threshold and rsi_now > self.threshold
        return False


class RSIOverboughtStrategy(Strategy):
    """Buy when RSI crosses below overbought threshold (mean reversion)."""
    
    def __init__(self, threshold: float = 70.0):
        super().__init__(f"RSI_Overbought_{int(threshold)}")
        self.threshold = threshold
    
    def should_enter(self, idx: int, df: pd.DataFrame, indicators: pd.DataFrame) -> bool:
        if idx < 1:
            return False
        
        rsi_prev = indicators.iloc[idx - 1]["rsi"]
        rsi_now = indicators.iloc[idx]["rsi"]
        
        # RSI crosses below threshold (was above, now below)
        if pd.notna(rsi_prev) and pd.notna(rsi_now):
            return rsi_prev >= self.threshold and rsi_now < self.threshold
        return False


class SMAGoldenCrossStrategy(Strategy):
    """Buy when fast SMA crosses above slow SMA (golden cross)."""
    
    def __init__(self, fast: int = 7, slow: int = 21):
        super().__init__(f"SMA_Cross_{fast}x{slow}")
        self.fast = fast
        self.slow = slow
    
    def should_enter(self, idx: int, df: pd.DataFrame, indicators: pd.DataFrame) -> bool:
        if idx < 1:
            return False
        
        sma_fast_prev = indicators.iloc[idx - 1]["sma7"]
        sma_slow_prev = indicators.iloc[idx - 1]["sma21"]
        sma_fast_now = indicators.iloc[idx]["sma7"]
        sma_slow_now = indicators.iloc[idx]["sma21"]
        
        # Fast crosses above slow
        if all(pd.notna([sma_fast_prev, sma_slow_prev, sma_fast_now, sma_slow_now])):
            return sma_fast_prev <= sma_slow_prev and sma_fast_now > sma_slow_now
        return False


class SMAMomentumStrategy(Strategy):
    """Buy when price is above SMA and SMA is rising."""
    
    def __init__(self, period: int = 21):
        super().__init__(f"SMA_Momentum_{period}")
        self.period = period
    
    def should_enter(self, idx: int, df: pd.DataFrame, indicators: pd.DataFrame) -> bool:
        if idx < 1:
            return False
        
        price = df.iloc[idx]["close"]
        sma = indicators.iloc[idx]["sma21"]
        sma_prev = indicators.iloc[idx - 1]["sma21"]
        
        # Price above SMA and SMA rising
        if pd.notna(sma) and pd.notna(sma_prev):
            return price > sma and sma > sma_prev
        return False


class ComboRSI_SMAStrategy(Strategy):
    """Buy when RSI oversold AND price crosses above SMA."""
    
    def __init__(self, rsi_threshold: float = 40.0):
        super().__init__(f"Combo_RSI{int(rsi_threshold)}_SMA")
        self.rsi_threshold = rsi_threshold
    
    def should_enter(self, idx: int, df: pd.DataFrame, indicators: pd.DataFrame) -> bool:
        if idx < 1:
            return False
        
        rsi = indicators.iloc[idx]["rsi"]
        price = df.iloc[idx]["close"]
        price_prev = df.iloc[idx - 1]["close"]
        sma = indicators.iloc[idx]["sma7"]
        sma_prev = indicators.iloc[idx - 1]["sma7"]
        
        # RSI oversold AND price crosses above fast SMA
        if all(pd.notna([rsi, price, price_prev, sma, sma_prev])):
            rsi_oversold = rsi < self.rsi_threshold
            price_crosses = price_prev <= sma_prev and price > sma
            return rsi_oversold and price_crosses
        return False


def backtest_strategy(
    strategy: Strategy,
    df: pd.DataFrame,
    indicators: pd.DataFrame,
    stop_ticks: int = 8,
    rr: float = 2.0,
    tick_size: float = 0.25,
    contract_multiplier: float = 5.0,
    max_bars: int = 500,
    cooldown_bars: int = 10,
) -> Dict:
    """Backtest a strategy with TP/SL exits."""
    prices = df["close"].values
    
    results = {
        "trades": [],
        "wins": 0,
        "losses": 0,
        "timeouts": 0,
        "total_pnl_ticks": 0.0,
        "total_pnl_usd": 0.0,
    }
    
    in_trade = False
    cooldown_until = -1
    
    for idx in range(len(df)):
        # Skip if in cooldown
        if idx < cooldown_until:
            continue
        
        # Check for entry signal
        if not in_trade and strategy.should_enter(idx, df, indicators):
            entry_price = prices[idx]
            
            # Simulate trade
            outcome, bars_held, pnl_ticks, exit_price = simulate_trade(
                prices, idx, entry_price, stop_ticks, rr, tick_size, max_bars
            )
            
            pnl_usd = pnl_ticks * tick_size * contract_multiplier
            
            results["trades"].append({
                "entry_idx": idx,
                "entry_price": entry_price,
                "exit_idx": idx + bars_held,
                "exit_price": exit_price,
                "bars_held": bars_held,
                "outcome": outcome,
                "pnl_ticks": pnl_ticks,
                "pnl_usd": pnl_usd,
            })
            
            if outcome == "TP":
                results["wins"] += 1
            elif outcome == "SL":
                results["losses"] += 1
            else:
                results["timeouts"] += 1
            
            results["total_pnl_ticks"] += pnl_ticks
            results["total_pnl_usd"] += pnl_usd
            
            # Set cooldown to avoid overlapping trades
            cooldown_until = idx + bars_held + cooldown_bars
    
    return results


def print_results(strategy_name: str, results: Dict):
    """Pretty print backtest results."""
    trades = results["trades"]
    wins = results["wins"]
    losses = results["losses"]
    timeouts = results["timeouts"]
    total_trades = wins + losses  # Exclude timeouts
    
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    avg_pnl_ticks = results["total_pnl_ticks"] / len(trades) if trades else 0.0
    avg_pnl_usd = results["total_pnl_usd"] / len(trades) if trades else 0.0
    
    avg_bars_tp = np.mean([t["bars_held"] for t in trades if t["outcome"] == "TP"]) if wins > 0 else 0
    avg_bars_sl = np.mean([t["bars_held"] for t in trades if t["outcome"] == "SL"]) if losses > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*70}")
    print(f"Total Trades:   {len(trades):4d}")
    print(f"  TP:           {wins:4d} ({wins/len(trades)*100 if trades else 0:5.1f}%)")
    print(f"  SL:           {losses:4d} ({losses/len(trades)*100 if trades else 0:5.1f}%)")
    print(f"  Timeout:      {timeouts:4d} ({timeouts/len(trades)*100 if trades else 0:5.1f}%)")
    print(f"\nMetrics:")
    print(f"  Win Rate:          {win_rate*100:5.1f}% (excludes timeouts)")
    print(f"  Total PnL:         {results['total_pnl_usd']:+10.2f} USD")
    print(f"  Avg PnL/Trade:     {avg_pnl_usd:+8.2f} USD ({avg_pnl_ticks:+6.2f} ticks)")
    print(f"  Avg Bars to TP:    {avg_bars_tp:6.1f} mins")
    print(f"  Avg Bars to SL:    {avg_bars_sl:6.1f} mins")
    
    return {
        "strategy": strategy_name,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl_usd": results["total_pnl_usd"],
        "avg_pnl_usd": avg_pnl_usd,
        "avg_pnl_ticks": avg_pnl_ticks,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest indicator strategies")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV with price data",
    )
    parser.add_argument(
        "--stop-ticks",
        type=int,
        default=8,
        help="Stop loss in ticks (default: 8 = 2 points)",
    )
    parser.add_argument(
        "--rr",
        type=float,
        default=2.0,
        help="Risk/reward ratio (default: 2.0)",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=0.25,
        help="Tick size for MES (default: 0.25)",
    )
    parser.add_argument(
        "--contract-multiplier",
        type=float,
        default=5.0,
        help="Contract multiplier for MES (default: 5.0)",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=500,
        help="Max bars to hold before timeout (default: 500)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=10,
        help="Cooldown bars after trade exit (default: 10)",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    if "close" not in df.columns:
        raise ValueError("CSV must have a 'close' column")
    
    print(f"Loaded {len(df):,} bars of price data")
    
    # Compute indicators
    print("Computing indicators (RSI, SMA7, SMA21)...")
    indicators = compute_indicators(df)
    
    # Define strategies to test
    strategies = [
        RSIOversoldStrategy(threshold=30),
        RSIOversoldStrategy(threshold=40),
        RSIOverboughtStrategy(threshold=70),
        RSIOverboughtStrategy(threshold=60),
        SMAGoldenCrossStrategy(fast=7, slow=21),
        SMAMomentumStrategy(period=21),
        ComboRSI_SMAStrategy(rsi_threshold=40),
    ]
    
    # Run backtests
    print(f"\n{'='*70}")
    print(f"Backtesting {len(strategies)} strategies")
    print(f"TP/SL: {args.stop_ticks} ticks stop ({args.stop_ticks * args.tick_size:.2f} pts), "
          f"{args.rr}:1 RR ({args.stop_ticks * args.rr * args.tick_size:.2f} pts target)")
    print(f"{'='*70}")
    
    all_results = []
    
    for strategy in strategies:
        results = backtest_strategy(
            strategy,
            df,
            indicators,
            stop_ticks=args.stop_ticks,
            rr=args.rr,
            tick_size=args.tick_size,
            contract_multiplier=args.contract_multiplier,
            max_bars=args.max_bars,
            cooldown_bars=args.cooldown,
        )
        
        summary = print_results(strategy.name, results)
        all_results.append(summary)
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Strategy Comparison")
    print(f"{'='*70}\n")
    
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values("total_pnl_usd", ascending=False)
    
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("Interpretation:")
    print(f"{'='*70}")
    
    best = summary_df.iloc[0]
    print(f"\nBest Strategy: {best['strategy']}")
    print(f"  Win Rate: {best['win_rate']*100:.1f}%")
    print(f"  Total PnL: ${best['total_pnl_usd']:,.2f}")
    print(f"  Avg PnL/Trade: ${best['avg_pnl_usd']:+.2f}")
    
    # Calculate break-even win rate
    breakeven_wr = 1.0 / (1.0 + args.rr)
    print(f"\nBreak-even win rate: {breakeven_wr*100:.1f}%")
    
    profitable_strategies = summary_df[summary_df["total_pnl_usd"] > 0]
    
    if len(profitable_strategies) > 0:
        print(f"✓ {len(profitable_strategies)}/{len(strategies)} strategies are profitable")
        print(f"\nThese indicators may have predictive power for your RL model!")
    else:
        print(f"✗ No strategies are profitable with these parameters")
        print(f"\nThis suggests:")
        print(f"  - Random entries might be as good as simple indicators")
        print(f"  - Your RL model needs to learn complex patterns")
        print(f"  - Consider testing different TP/SL settings or time filters (RTH only)")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

