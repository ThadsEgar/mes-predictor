# scripts/train_from_csv.py
import pandas as pd
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

CSV = "datasets/mes_finrl_ready.csv"  # your cleaned file

# Load data
df = pd.read_csv(CSV, parse_dates=["timestamp"])
# Optional: unify ticker
# df["tic"] = "MES"

# Minimal arrays for env: close-only price, no tech indicators, zero turbulence
price_array = df[["close"]].values
tech_array = np.zeros((len(df), 0))  # no indicators
turbulence_array = np.zeros(len(df))

env_config = {
    "price_array": price_array,
    "tech_array": tech_array,
    "turbulence_array": turbulence_array,
    "if_train": True,
}

env = StockTradingEnv(config=env_config)
agent = DRLAgent_sb3(env=env)
model = agent.get_model("ppo")  # or "sac", "td3", etc.
trained = agent.train_model(model=model, tb_log_name="ppo", total_timesteps=200000)
trained.save("./models/ppo_mes")
print("Training done. Model saved to ./models/ppo_mes")