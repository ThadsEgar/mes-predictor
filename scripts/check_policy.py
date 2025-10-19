#!/usr/bin/env python3
"""Quick check of model's action probabilities"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from finrl.meta.env_stock_trading.env_dense_trading import DenseRewardTradingEnv
from scripts.utils import compute_indicators

# Load data
df = pd.read_csv("datasets/mes_finrl_ready_front.csv").tail(5000).reset_index(drop=True)
price_array = df[["close"]].values.flatten()
tech_array = compute_indicators(df).values.astype(float)

# Create env
env = DenseRewardTradingEnv(
    price_array=price_array,
    tech_array=tech_array,
    initial_capital=1e6,
    tick_size=0.25,
    contract_multiplier=5.0,
    transaction_cost_bps=2.0,
    inactivity_penalty=0.0,
)

# Load model
model = PPO.load("models/ppo_dense_metrics_ent0.3")

# Check first 100 steps
obs, _ = env.reset()
print("Checking action probabilities for first 100 steps:")
print(f"{'Step':<6} {'Action':<8} {'P(hold)':<10} {'P(buy/sell)':<12} {'Position':<10}")
print("-" * 60)

for i in range(100):
    # Get action probabilities
    action_probs = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0]).distribution.probs.detach().numpy()[0]
    
    # Get action
    action, _ = model.predict(obs, deterministic=True)
    
    # Step
    obs, _, terminated, truncated, _ = env.step(action)
    
    print(f"{i:<6} {action:<8} {action_probs[0]:<10.4f} {action_probs[1]:<12.4f} {env.position:<10}")
    
    if terminated or truncated:
        break

print(f"\nSummary:")
print(f"Actions: hold={np.sum([1 for i in range(min(100, i+1))])}, buy/sell=0")
print(f"Avg P(hold): {np.mean([action_probs[0]]):.4f}")
print(f"Model is {'STUCK' if action_probs[0] > 0.99 else 'exploring'}")

