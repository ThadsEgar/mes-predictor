"""Evaluate trained PPO models and plot equity curves."""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from typing import List, Tuple

from stable_baselines3 import PPO
from finrl.meta.env_stock_trading.env_dense_trading import DenseRewardTradingEnv
from scripts.utils import compute_indicators


def resolve_model_paths(model: str | None, models_dir: str | None, pattern: str) -> List[str]:
    if model:
        p = model if model.endswith(".zip") else f"{model}.zip"
        return [p]
    if not models_dir:
        models_dir = "./models"
    paths = sorted(glob(os.path.join(models_dir, pattern)))
    return paths


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def evaluate_one(
    model_path: str,
    csv_path: str,
    last_n: int | None,
    out_png: str,
    out_csv: str,
) -> Tuple[float, float, int, int]:
    # Load data
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if last_n is not None and last_n > 0:
        needed_rows = min(len(df), last_n + 1)
        df = df.tail(needed_rows).reset_index(drop=True)

    price_array = df[["close"]].values
    tech_df = compute_indicators(df)
    tech_array = tech_df.values.astype(float)

    # Create environment (matching training setup)
    env = DenseRewardTradingEnv(
        price_array=price_array,
        tech_array=tech_array,
        initial_capital=1e6,
        tick_size=0.25,
        contract_multiplier=5.0,
        transaction_cost_bps=2.0,  # Match training
        inactivity_penalty=0.0,  # No penalty during eval
    )

    # Load model
    model = PPO.load(model_path)
    
    # Run evaluation
    obs, _ = env.reset()
    assets = [env.cash + env.position * price_array[0, 0] * env.contract_multiplier]
    
    trade_logs = []
    done = False
    step_idx = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log step
        price = env.price_array[env.day - 1]
        total_asset = env.cash + env.position * price * env.contract_multiplier
        assets.append(float(total_asset))
        
        trade_logs.append({
            "step": step_idx,
            "action": int(action),
            "position": env.position,
            "price": float(price),
            "cash": float(env.cash),
            "total_asset": float(total_asset),
        })
        
        done = bool(terminated or truncated)
        step_idx += 1

    # Extract final stats
    wins = env.wins
    losses = env.losses
    
    # Prepare outputs
    dates = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    if len(assets) > len(dates):
        assets = assets[:len(dates)]
    elif len(assets) < len(dates):
        dates = dates[:len(assets)]

    account_value = pd.DataFrame({"date": dates, "account_value": assets})

    # Save CSVs
    ensure_dir(out_csv)
    account_value.to_csv(out_csv, index=False)
    
    trades_out_csv = out_csv.replace("account_value_eval.csv", "trades_eval.csv")
    trades_df = pd.DataFrame(trade_logs)
    trades_df.to_csv(trades_out_csv, index=False)

    # Plot equity curve
    ensure_dir(out_png)
    plt.figure(figsize=(12, 4))
    plt.plot(account_value["date"], account_value["account_value"], label="Equity")
    
    # Mark entries
    entries = [i for i, log in enumerate(trade_logs) if log["position"] == 1 and (i == 0 or trade_logs[i-1]["position"] == 0)]
    if entries:
        entry_dates = [dates[min(i, len(dates)-1)] for i in entries]
        entry_vals = [account_value["account_value"].iloc[min(i, len(account_value)-1)] for i in entries]
        plt.scatter(entry_dates, entry_vals, marker="^", color="g", label="Entry", s=50, alpha=0.7)
    
    plt.title(f"Equity Curve (W/L: {wins}/{losses})")
    plt.xlabel("Date")
    plt.ylabel("Account Value ($)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    initial_asset = float(account_value["account_value"].iloc[0])
    final_asset = float(account_value["account_value"].iloc[-1])
    final_return = final_asset / initial_asset - 1.0 if initial_asset != 0 else float("nan")
    
    return final_asset, final_return, wins, losses


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch evaluate PPO models and plot equity curves")
    parser.add_argument("--csv", default="datasets/mes_finrl_ready_front.csv",
                        help="Input data CSV (must contain 'timestamp' and 'close')")
    parser.add_argument("--model", default=None, help="Single model base path ('.zip' auto)")
    parser.add_argument("--models-dir", default="./models", help="Directory to scan for models")
    parser.add_argument("--pattern", default="*_ppo_*.zip", help="Glob pattern under models-dir")
    parser.add_argument("--last-n", type=int, default=None, help="Evaluate only last N timesteps")
    parser.add_argument("--out-dir", default="results", help="Output directory for CSVs")
    parser.add_argument("--fig-dir", default="figs", help="Output directory for PNGs")

    args = parser.parse_args()

    model_paths = resolve_model_paths(args.model, args.models_dir, args.pattern)
    if not model_paths:
        raise FileNotFoundError("No model files found. Provide --model or --models-dir/--pattern.")

    summary = []
    for mp in model_paths:
        base = os.path.splitext(os.path.basename(mp))[0]
        out_csv = os.path.join(args.out_dir, f"{base}_account_value_eval.csv")
        out_png = os.path.join(args.fig_dir, f"{base}_equity_curve.png")
        
        print(f"Evaluating {mp} ...")
        try:
            final_asset, final_return, wins, losses = evaluate_one(
                model_path=mp,
                csv_path=args.csv,
                last_n=args.last_n,
                out_png=out_png,
                out_csv=out_csv,
            )
            summary.append({
                "model": base,
                "final_asset": final_asset,
                "final_return": final_return,
                "wins": wins,
                "losses": losses,
                "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0.0,
            })
            print(f"  → return={final_return:.4f} | W/L={wins}/{losses} | Saved: {out_png}")
        except Exception as e:
            print(f"  → ERROR: {e}")
            summary.append({
                "model": base,
                "final_asset": 0.0,
                "final_return": 0.0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "error": str(e),
            })

    ensure_dir(os.path.join(args.out_dir, "summary.csv"))
    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    print(f"\nSummary written to {os.path.join(args.out_dir, 'summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
