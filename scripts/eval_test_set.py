#!/usr/bin/env python3
"""
Evaluate model on UNSEEN test set (last portion of data).
This gives you realistic out-of-sample performance.
"""

import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from finrl.meta.env_stock_trading.env_dense_trading import DenseRewardTradingEnv
from scripts.utils import compute_indicators
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="datasets/mes_finrl_ready_front.csv")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--total-bars", type=int, default=350000, help="Total dataset size")
    parser.add_argument("--train-size", type=float, default=0.8, help="Fraction used for training (default: 0.8)")
    parser.add_argument("--save-dir", default="eval_results", help="Directory to save results")
    parser.add_argument("--transaction-cost", type=float, default=0.5, help="Transaction cost in bps")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load full dataset
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)

    # Get the slice
    if args.total_bars < len(df):
        df = df.tail(args.total_bars).reset_index(drop=True)

    # Calculate split point
    train_bars = int(len(df) * args.train_size)
    test_bars = len(df) - train_bars

    print(f"\n{'='*60}")
    print(f"TRAIN/TEST SPLIT")
    print(f"{'='*60}")
    print(f"Total bars: {len(df):,}")
    print(f"Train bars: {train_bars:,} (first {args.train_size*100:.0f}%)")
    print(f"Test bars: {test_bars:,} (last {(1-args.train_size)*100:.0f}%) ← UNSEEN DATA")
    print(f"{'='*60}")

    # Extract TEST set only
    df_test = df.iloc[train_bars:].reset_index(drop=True)
    print(f"\nEvaluating on TEST SET ONLY: {len(df_test):,} bars")
    print(f"This data was NOT seen during training!")

    # Prepare test data
    price_array = df_test[["close"]].values.flatten()
    tech_array = compute_indicators(df_test).values.astype(float)

    # Create environment
    env = DenseRewardTradingEnv(
        price_array=price_array,
        tech_array=tech_array,
        tick_size=0.25,
        contract_multiplier=5.0,
        transaction_cost_bps=args.transaction_cost,
        inactivity_penalty=0.0,
        max_hold_bars=240,
        holding_loss_penalty=True,
        grace_period_bars=45,
        emergency_stop_loss=-50.0,
    )

    # Load model
    print(f"\nLoading model from {args.model}...")
    model = PPO.load(args.model)

    # Load VecNormalize stats
    vec_norm_path = f"{args.model}_vecnormalize.pkl"
    if os.path.exists(vec_norm_path):
        print(f"Loading normalization stats from {vec_norm_path}...")
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("No normalization stats found")
        env = DummyVecEnv([lambda: env])

    # Run evaluation
    print("\nRunning evaluation on UNSEEN test data...")
    obs = env.reset()

    # Track everything
    actions = []
    prices = []
    cumulative_pnl = [0.0]
    positions = []
    trades = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # Get underlying env
        base_env = env.envs[0]
        if hasattr(base_env, 'env'):
            base_env = base_env.env

        current_price = base_env.price_array[base_env.day]
        current_position = base_env.position

        # Step
        obs, reward, done, info = env.step(action)
        if isinstance(done, np.ndarray):
            done = done[0]
        if isinstance(info, list):
            info = info[0]

        # Record
        actions.append(int(action))
        prices.append(current_price)

        updated_base_env = env.envs[0]
        if hasattr(updated_base_env, 'env'):
            updated_base_env = updated_base_env.env
        cumulative_pnl.append(updated_base_env.total_pnl)
        positions.append(current_position)

        # Record trades
        if "action" in info:
            trade = {
                "day": base_env.day - 1,
                "type": info["action"],
                "price": info.get("entry_price", info.get("exit_price", current_price)),
                "pnl": info.get("trade_pnl", 0),
                "bars_held": info.get("bars_held", 0),
            }
            trades.append(trade)

    # Print summary
    print("\n" + "="*60)
    print("TEST SET PERFORMANCE (UNSEEN DATA)")
    print("="*60)
    print(f"Total PnL: ${base_env.total_pnl:,.2f}")
    print(f"\nTotal Trades: {base_env.trades}")
    print(f"Wins: {base_env.wins} ({base_env.wins/base_env.trades*100:.1f}%)" if base_env.trades > 0 else "Wins: 0")
    print(f"Losses: {base_env.losses} ({base_env.losses/base_env.trades*100:.1f}%)" if base_env.trades > 0 else "Losses: 0")
    print(f"Avg Bars Held: {base_env.total_bars_held/base_env.trades:.1f}" if base_env.trades > 0 else "Avg Bars Held: N/A")

    # Create plot
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'Test Set - Total PnL: ${base_env.total_pnl:,.0f}',
            'Cumulative PnL Over Time',
            'Position Over Time'
        ),
        row_heights=[0.4, 0.3, 0.3]
    )

    # Plot price
    fig.add_trace(
        go.Scatter(
            x=list(range(len(prices))),
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='blue', width=1),
        ),
        row=1, col=1
    )

    # Add trades
    buy_trades = [t for t in trades if t["type"] == "buy"]
    if buy_trades:
        fig.add_trace(
            go.Scatter(
                x=[t["day"] for t in buy_trades],
                y=[t["price"] for t in buy_trades],
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=10, symbol='triangle-up'),
            ),
            row=1, col=1
        )

    sell_trades = [t for t in trades if t["type"] in ["sell", "sell_forced", "sell_emergency_stop"]]
    if sell_trades:
        sell_colors = ['green' if t["pnl"] > 0 else 'red' for t in sell_trades]
        fig.add_trace(
            go.Scatter(
                x=[t["day"] for t in sell_trades],
                y=[t["price"] for t in sell_trades],
                mode='markers',
                name='Sell',
                marker=dict(color=sell_colors, size=10, symbol='triangle-down'),
            ),
            row=1, col=1
        )

    # PnL curve
    fig.add_trace(
        go.Scatter(
            x=list(range(len(cumulative_pnl))),
            y=cumulative_pnl,
            mode='lines',
            name='Cumulative PnL',
            line=dict(color='green', width=2),
            fill='tozeroy',
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)

    # Position
    fig.add_trace(
        go.Scatter(
            x=list(range(len(positions))),
            y=positions,
            mode='lines',
            name='Position',
            line=dict(color='blue', width=2),
            fill='tozeroy',
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified',
        title_text=f"Test Set Evaluation - {test_bars:,} UNSEEN bars"
    )

    # Save
    html_path = os.path.join(args.save_dir, f"test_set_eval_{timestamp}.html")
    fig.write_html(html_path)
    print(f"\nTest set plot saved to: {html_path}")

    # Save trades
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_path = os.path.join(args.save_dir, f"test_set_trades_{timestamp}.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"Test set trades saved to: {trades_path}")

    # Save metrics
    metrics = {
        "dataset": "TEST_SET_UNSEEN",
        "test_bars": test_bars,
        "total_pnl": base_env.total_pnl,
        "total_trades": base_env.trades,
        "wins": base_env.wins,
        "losses": base_env.losses,
        "win_rate": base_env.wins/base_env.trades*100 if base_env.trades > 0 else 0,
        "avg_bars_held": base_env.total_bars_held/base_env.trades if base_env.trades > 0 else 0,
        "transaction_cost_bps": args.transaction_cost,
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(args.save_dir, f"test_set_metrics_{timestamp}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Test set metrics saved to: {metrics_path}")

    print("\n" + "="*60)
    print("✓ Evaluation complete on UNSEEN test data!")
    print("="*60)


if __name__ == "__main__":
    main()
