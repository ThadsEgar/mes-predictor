#!/usr/bin/env python3
"""
Evaluation script for dense reward models.
Shows detailed trades, metrics, and plots.
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
    parser.add_argument("--model", default="models/ppo_dense")
    parser.add_argument("--train-slice", type=int, default=5000, help="Use same slice as training")
    parser.add_argument("--save-dir", default="eval_results", help="Directory to save results")
    parser.add_argument("--no-costs", action="store_true", help="Disable transaction costs")
    parser.add_argument("--transaction-cost", type=float, default=0.5, help="Transaction cost in bps (default: 0.5)")
    parser.add_argument("--max-hold-bars", type=int, default=60, help="Maximum bars to hold position (default: 60)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data (same as training)
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    
    if args.train_slice and args.train_slice < len(df):
        df = df.tail(args.train_slice).reset_index(drop=True)
    
    print(f"Evaluating on {len(df)} bars (same as training)")
    
    # Prepare data
    price_array = df[["close"]].values.flatten()
    tech_array = compute_indicators(df).values.astype(float)
    
    # Create environment (match training params)
    transaction_cost = 0.0 if args.no_costs else args.transaction_cost
    env = DenseRewardTradingEnv(
        price_array=price_array,
        tech_array=tech_array,
        tick_size=0.25,
        contract_multiplier=5.0,
        transaction_cost_bps=transaction_cost,
        inactivity_penalty=0.0,  # No penalty during eval
        max_hold_bars=args.max_hold_bars,
        holding_loss_penalty=True,  # Match training
        grace_period_bars=45,  # Match training
        emergency_stop_loss=-50.0,  # Match training
    )
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = PPO.load(args.model)
    
    # Check if VecNormalize stats exist and load them
    vec_norm_path = f"{args.model}_vecnormalize.pkl"
    if os.path.exists(vec_norm_path):
        print(f"Loading normalization stats from {vec_norm_path}...")
        # Wrap env in DummyVecEnv first (VecNormalize needs vectorized env)
        env = DummyVecEnv([lambda: env])
        # Load normalization stats
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False  # Don't update stats during eval
        env.norm_reward = False  # Don't normalize rewards during eval
    else:
        print("No normalization stats found, using raw observations")
        # Still need to wrap for consistency
        env = DummyVecEnv([lambda: env])
    
    # Run evaluation
    print("Running evaluation...")
    obs = env.reset()
    
    # Track everything
    actions = []
    prices = []
    cumulative_pnl = [0.0]  # Track cumulative PnL over time
    positions = []
    trades = []
    
    done = False
    while not done:
        # Get action
        action, _ = model.predict(obs, deterministic=True)
        
        # Get underlying env for data access
        base_env = env.envs[0]
        if hasattr(base_env, 'env'):  # If wrapped in VecNormalize
            base_env = base_env.env
        
        # Store state before step
        current_price = base_env.price_array[base_env.day]
        current_position = base_env.position
        
        # Step
        obs, reward, done, info = env.step(action)
        if isinstance(done, np.ndarray):
            done = done[0]  # Extract from array
        if isinstance(info, list):
            info = info[0]  # Extract from list
        
        # Record
        actions.append(int(action))
        prices.append(current_price)
        # Get updated PnL after step
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
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total PnL: ${base_env.total_pnl:,.2f}")
    print(f"\nTotal Trades: {base_env.trades}")
    print(f"Wins: {base_env.wins} ({base_env.wins/base_env.trades*100:.1f}%)" if base_env.trades > 0 else "Wins: 0")
    print(f"Losses: {base_env.losses} ({base_env.losses/base_env.trades*100:.1f}%)" if base_env.trades > 0 else "Losses: 0")
    print(f"Avg Bars Held: {base_env.total_bars_held/base_env.trades:.1f}" if base_env.trades > 0 else "Avg Bars Held: N/A")
    print(f"Win Rate (no costs): {base_env.wins_no_cost/(base_env.wins_no_cost+base_env.losses_no_cost)*100:.1f}%" 
          if (base_env.wins_no_cost+base_env.losses_no_cost) > 0 else "Win Rate (no costs): N/A")
    
    # Create interactive Plotly plot with zoom/pan capabilities
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'MES Price and Trades - Total PnL: ${base_env.total_pnl:,.0f}',
            'Cumulative PnL Over Time',
            'Position Over Time'
        ),
        row_heights=[0.4, 0.3, 0.3]
    )

    # Plot 1: Price line
    fig.add_trace(
        go.Scatter(
            x=list(range(len(prices))),
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='blue', width=1),
            hovertemplate='Bar: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add buy trades
    buy_trades = [t for t in trades if t["type"] == "buy"]
    if buy_trades:
        fig.add_trace(
            go.Scatter(
                x=[t["day"] for t in buy_trades],
                y=[t["price"] for t in buy_trades],
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                hovertemplate='<b>BUY</b><br>Bar: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

    # Add sell trades (color by profit) - includes "sell", "sell_forced", and "sell_emergency_stop"
    sell_trades = [t for t in trades if t["type"] in ["sell", "sell_forced", "sell_emergency_stop"]]
    if sell_trades:
        sell_colors = ['green' if t["pnl"] > 0 else 'red' for t in sell_trades]
        sell_text = [f"PnL: ${t['pnl']:.0f}, Held: {t['bars_held']} bars" for t in sell_trades]

        fig.add_trace(
            go.Scatter(
                x=[t["day"] for t in sell_trades],
                y=[t["price"] for t in sell_trades],
                mode='markers',
                name='Sell',
                marker=dict(color=sell_colors, size=10, symbol='triangle-down'),
                text=sell_text,
                hovertemplate='<b>SELL</b><br>Bar: %{x}<br>Price: $%{y:.2f}<br>%{text}<extra></extra>'
            ),
            row=1, col=1
        )

    # Plot 2: Cumulative PnL
    fig.add_trace(
        go.Scatter(
            x=list(range(len(cumulative_pnl))),
            y=cumulative_pnl,
            mode='lines',
            name='Cumulative PnL',
            line=dict(color='green', width=2),
            fill='tozeroy',
            hovertemplate='Bar: %{x}<br>PnL: $%{y:,.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Add break-even line
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)

    # Plot 3: Position
    fig.add_trace(
        go.Scatter(
            x=list(range(len(positions))),
            y=positions,
            mode='lines',
            name='Position',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            hovertemplate='Bar: %{x}<br>Position: %{y}<extra></extra>'
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative PnL ($)", row=2, col=1)
    fig.update_yaxes(title_text="Position (Contracts)", row=3, col=1, range=[-0.1, 1.1])

    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified',
        title_text=f"Trading Evaluation - {len(prices)} bars, {len(trades)} trades"
    )

    # Save interactive HTML (open this in browser to zoom/pan)
    html_path = os.path.join(args.save_dir, f"eval_plot_{timestamp}.html")
    fig.write_html(html_path)
    print(f"\nInteractive plot saved to: {html_path}")
    print("  -> Open in browser to zoom/pan through all data points!")

    # Also save static PNG for quick viewing
    try:
        png_path = os.path.join(args.save_dir, f"eval_plot_{timestamp}.png")
        fig.write_image(png_path, width=1600, height=900)
        print(f"Static PNG saved to: {png_path}")
    except Exception as e:
        print(f"Note: Could not save PNG (install kaleido for PNG export): {e}")
    
    # Save trade log
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_path = os.path.join(args.save_dir, f"trades_{timestamp}.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"Trade log saved to: {trades_path}")
        
        # Print first few trades
        print("\nFirst 10 trades:")
        print(trades_df.head(10).to_string())
    
    # Get base environment for metrics
    base_env = env.envs[0]
    if hasattr(base_env, 'env'):  # If wrapped in VecNormalize
        base_env = base_env.env
    
    # Save detailed metrics
    metrics = {
        "total_pnl": base_env.total_pnl,
        "total_trades": base_env.trades,
        "wins": base_env.wins,
        "losses": base_env.losses,
        "win_rate": base_env.wins/base_env.trades*100 if base_env.trades > 0 else 0,
        "wins_no_cost": base_env.wins_no_cost,
        "losses_no_cost": base_env.losses_no_cost,
        "win_rate_no_cost": base_env.wins_no_cost/(base_env.wins_no_cost+base_env.losses_no_cost)*100 
                           if (base_env.wins_no_cost+base_env.losses_no_cost) > 0 else 0,
        "avg_bars_held": base_env.total_bars_held/base_env.trades if base_env.trades > 0 else 0,
        "buy_actions": sum(actions),
        "hold_actions": len(actions) - sum(actions),
        "buy_rate": sum(actions) / len(actions) * 100,
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(args.save_dir, f"metrics_{timestamp}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
