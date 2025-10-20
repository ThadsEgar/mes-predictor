#!/usr/bin/env python3
"""
Quick training script for dense reward environment.

This is a simplified version focused on the new dense reward approach.
"""

import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.monitor import Monitor
from finrl.meta.env_stock_trading.env_dense_trading import DenseRewardTradingEnv
from finrl.meta.callbacks.dense_metrics import DenseEnvStatsCallback
from finrl.meta.callbacks.entropy_decay import EntropyDecayCallback
from finrl.meta.callbacks.lr_decay import LearningRateDecayCallback
from scripts.utils import compute_indicators
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="datasets/mes_finrl_ready_front.csv")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-decay", action="store_true", help="Enable linear learning rate decay")
    parser.add_argument("--lr-end", type=float, default=1e-6, help="Final learning rate")
    parser.add_argument("--lr-decay-steps", type=int, default=None, help="Steps to decay LR over (default: total_timesteps)")
    parser.add_argument("--ent-coef", type=float, default=0.1)  # Starting entropy
    parser.add_argument("--ent-decay", action="store_true", help="Enable linear entropy decay")
    parser.add_argument("--ent-end", type=float, default=0.01, help="Final entropy coefficient")
    parser.add_argument("--ent-decay-steps", type=int, default=None, help="Steps to decay over (default: total_timesteps)")
    parser.add_argument("--train-slice", type=int, default=50000)
    parser.add_argument("--name", default="ppo_dense")
    parser.add_argument("--load-model", type=str, default=None, help="Path to existing model to continue training")
    parser.add_argument("--transaction-cost", type=float, default=0.5, help="Transaction cost in bps")
    args = parser.parse_args()
    
    # Load and prepare data
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Use last N bars for training
    if args.train_slice and args.train_slice < len(df):
        df = df.tail(args.train_slice).reset_index(drop=True)
    
    print(f"Training on {len(df)} bars")
    
    # Create env factory
    def make_env(rank: int):
        def _init():
            price_array = df[["close"]].values
            tech_array = compute_indicators(df).values.astype(float)
            
            env = DenseRewardTradingEnv(
                price_array=price_array,
                tech_array=tech_array,
                tick_size=0.25,
                contract_multiplier=5.0,
                transaction_cost_bps=args.transaction_cost,
                inactivity_penalty=0.0,  # No inactivity penalty
                max_hold_bars=240,  # Maximum 4 hours for day trading
                holding_loss_penalty=True,  # Penalize holding unrealized losses
                grace_period_bars=45,  # 45 minute grace period before penalty
                emergency_stop_loss=-50.0,  # Force exit if loss exceeds $50
            )
            return Monitor(env)
        return _init
    
    # Create vectorized env
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])
    env = VecMonitor(env)
    
    # Add VecNormalize for automatic observation normalization
    env = VecNormalize(
        env,
        norm_obs=True,       # Normalize observations (prices, indicators)
        norm_reward=False,   # Keep rewards as-is (our exponential scaling)
        clip_obs=10.0,       # Clip normalized obs to [-10, 10]
        gamma=0.99,          # Not used since norm_reward=False
    )
    
    # Create model with sensible defaults for dense rewards
    from torch import nn
    
    # Dynamic batch size based on n_envs
    n_steps = 2048  # Steps per env before update
    total_rollout_size = args.n_envs * n_steps
    
    # Batch size rules:
    # - Should divide evenly into rollout size
    # - Larger = faster (fewer gradient steps) but less stable
    # - For 32 envs: can go quite large
    if args.n_envs >= 32:
        batch_size = min(total_rollout_size // 4, 8192)  # Big batches OK
        n_epochs = 3  # Fewer epochs needed with diverse data
    elif args.n_envs >= 16:
        batch_size = min(total_rollout_size // 8, 4096)
        n_epochs = 4
    else:
        batch_size = min(total_rollout_size // 16, 2048)
        n_epochs = 5
    
    print(f"\nEnvironment setup:")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  Steps per env: {n_steps}")
    print(f"  Total rollout: {total_rollout_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient steps per update: {(total_rollout_size * n_epochs) // batch_size}")
    print(f"  Transaction cost: {args.transaction_cost} bps")

    # Load existing model or create new one
    if args.load_model:
        print(f"\nLoading existing model from {args.load_model}...")
        model = PPO.load(args.load_model, env=env, tensorboard_log="runs/tensorboard")

        # Try to load VecNormalize stats if they exist
        vec_norm_path = f"{args.load_model}_vecnormalize.pkl"
        if os.path.exists(vec_norm_path):
            print(f"Loading normalization stats from {vec_norm_path}...")
            # Need to unwrap the current env and rewrap with loaded stats
            base_envs = env.venv
            env = VecNormalize.load(vec_norm_path, base_envs)
            env.training = True  # Enable training mode
            env.norm_reward = False  # Keep our reward scaling
            model.set_env(env)
        else:
            print("Warning: No normalization stats found, using current env normalization")

        print(f"Continuing training with:")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Entropy coef: {args.ent_coef}")
    else:
        print("\nCreating new model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.05,  # Stop update if policy changes too much (prevents collapse)
            policy_kwargs=dict(
                net_arch=[256, 128, 128],  # 3 layers, 256 neurons each
                activation_fn=nn.ReLU,
            ),
            tensorboard_log="runs/tensorboard",
            verbose=1,
        )
    
    print(f"\nTraining {args.name} for {args.timesteps:,} steps...")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Entropy coef: {args.ent_coef}")
    
    # Setup callbacks
    callbacks = [DenseEnvStatsCallback()]
    
    if args.lr_decay:
        lr_decay_steps = args.lr_decay_steps if args.lr_decay_steps else args.timesteps
        print(f"Learning rate decay: {args.learning_rate} → {args.lr_end} over {lr_decay_steps:,} steps")
        callbacks.append(LearningRateDecayCallback(
            lr_start=args.learning_rate,
            lr_end=args.lr_end,
            decay_steps=lr_decay_steps
        ))
    
    if args.ent_decay:
        ent_decay_steps = args.ent_decay_steps if args.ent_decay_steps else args.timesteps
        print(f"Entropy decay: {args.ent_coef} → {args.ent_end} over {ent_decay_steps:,} steps")
        callbacks.append(EntropyDecayCallback(
            ent_start=args.ent_coef,
            ent_end=args.ent_end,
            decay_steps=ent_decay_steps
        ))
    
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=args.name,
        callback=callbacks,
    )
    
    # Save model and normalization stats
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{args.name}")
    env.save(f"models/{args.name}_vecnormalize.pkl")
    print(f"\nSaved model to: models/{args.name}.zip")
    print(f"Saved normalization stats to: models/{args.name}_vecnormalize.pkl")
    
    env.close()


if __name__ == "__main__":
    main()
