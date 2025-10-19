# MES Predictor Training Scripts

## train_from_csv.py

This script trains a PPO agent to trade MES (Micro E-mini S&P 500) futures using a discrete action space (0=hold, 1=buy). The environment automatically exits positions at predefined TP/SL levels.

### Basic Usage

```bash
python scripts/train_from_csv.py \
  --csv datasets/mes_finrl_ready_front.csv \
  --total-timesteps 10000000
```

### Common Training Scenarios

#### 1. Quick Overfit Test (Sanity Check)
Test if the model can learn on a small dataset:
```bash
python scripts/train_from_csv.py \
  --csv datasets/mes_finrl_ready_front.csv \
  --train-slice-bars 5000 \
  --n-envs 1 \
  --total-timesteps 2000000 \
  --tb-log-dir runs/tensorboard \
  --ent-values 0.01
```

#### 2. Full Training with Multiple Environments
Train on larger dataset with parallel environments:
```bash
python scripts/train_from_csv.py \
  --csv datasets/mes_finrl_ready_front.csv \
  --train-slice-bars 50000 \
  --n-envs 8 --vec-backend subproc \
  --total-timesteps 50000000 \
  --overnight-penalty-bps 0.0 \
  --tb-log-dir runs/tensorboard \
  --sweep-ent --ent-values 0.05
```

#### 3. Entropy Coefficient Sweep
Test multiple entropy values to find optimal exploration:
```bash
python scripts/train_from_csv.py \
  --csv datasets/mes_finrl_ready_front.csv \
  --train-slice-bars 50000 \
  --n-envs 8 --vec-backend subproc \
  --total-timesteps 10000000 \
  --tb-log-dir runs/tensorboard \
  --sweep-ent --ent-values "0.01,0.02,0.05,0.1,0.2"
```

#### 3b. Learning Rate Sweep
Test multiple learning rates to find optimal training speed:
```bash
python scripts/train_from_csv.py \
  --csv datasets/mes_finrl_ready_front.csv \
  --train-slice-bars 50000 \
  --n-envs 8 --vec-backend subproc \
  --total-timesteps 10000000 \
  --tb-log-dir runs/tensorboard \
  --sweep-lr --lr-values "1e-6,3e-6,1e-5,3e-5,1e-4"
```

#### 3c. Combined Parameter Sweep
Test all combinations of entropy and learning rate:
```bash
python scripts/train_from_csv.py \
  --csv datasets/mes_finrl_ready_front.csv \
  --train-slice-bars 20000 \
  --n-envs 4 --vec-backend subproc \
  --total-timesteps 5000000 \
  --tb-log-dir runs/tensorboard \
  --sweep-ent --ent-values "0.01,0.05" \
  --sweep-lr --lr-values "1e-5,1e-4"
```
This will train 4 models (2 entropy × 2 learning rate combinations).

#### 4. Continue Training Existing Model
Resume training from a checkpoint:
```bash
python scripts/train_from_csv.py \
  --csv datasets/mes_finrl_ready_front.csv \
  --train-slice-bars 50000 \
  --n-envs 8 --vec-backend subproc \
  --continue-from ./models/ppo_ent_0.05 \
  --continue-timesteps 20000000 \
  --tb-log-dir runs/tensorboard
```

#### 5. Training with Validation Holdout
Reserve last portion of data for validation:
```bash
python scripts/train_from_csv.py \
  --csv datasets/mes_finrl_ready_front.csv \
  --exclude-last-n 8190 \
  --n-envs 8 --vec-backend subproc \
  --total-timesteps 50000000 \
  --tb-log-dir runs/tensorboard \
  --ent-values 0.05
```

### Command Line Arguments

#### Data Options
- `--csv`: Path to input CSV file (default: datasets/mes_finrl_ready_front.csv)
- `--train-slice-bars`: Use only last N bars for training (useful for quick tests)
- `--exclude-last-n`: Exclude last N rows for validation
- `--train-val-split`: Train/validation split ratio (default: 0.8)

#### Training Options
- `--total-timesteps`: Total training timesteps (default: 200000)
- `--continue-from`: Path to existing model to continue training
- `--continue-timesteps`: Additional timesteps when continuing (defaults to --total-timesteps)
- `--n-envs`: Number of parallel environments (default: 8)
- `--vec-backend`: "subproc" or "dummy" for vectorized envs (default: subproc)

#### Environment Options
- `--overnight-penalty-bps`: Penalty for holding across session boundary (default: 0.0)
- `--contract-multiplier`: Contract multiplier for MES (default: 1.0, use 5.0 for realistic USD)

#### Hyperparameter Sweep
- `--sweep-ent`: Enable entropy coefficient sweep
- `--ent-values`: Comma-separated entropy values (default: "0.0,0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.08,0.1")
- `--sweep-lr`: Enable learning rate sweep
- `--lr-values`: Comma-separated learning rate values (default: "1e-6,3e-6,1e-5,3e-5,1e-4,3e-4")

#### Logging
- `--tb-log-dir`: TensorBoard log directory (default: runs/tensorboard)

### Environment Details

The discrete trading environment has:
- **Actions**: 0 (hold), 1 (buy 1 contract if flat)
- **Automatic TP/SL**: 
  - Stop Loss: 4 ticks (1 point) = $20 loss per contract
  - Take Profit: 8 ticks (2 points) = $40 profit per contract (2:1 risk/reward)
- **Rewards**:
  - TP hit: +1.0
  - SL hit: -1.0
  - Invalid buy while holding: -0.5
  - Inactivity penalty: -0.001 per step after 100 bars flat
- **Observations**: [cash_ratio, position, entry_price_diff, time_since_exit] + technical indicators

### Monitoring Training

View training progress in TensorBoard:
```bash
tensorboard --logdir runs/tensorboard
```

Key metrics to watch:
- `rollout/ep_rew_mean`: Average episode reward (should increase)
- `rollout/buy_rate`: Percentage of buy actions (should stabilize between 0.1-0.3)
- `rollout/win_rate`: Win percentage (target > 33% for 2:1 RR)
- `rollout/win_loss_ratio`: Wins/Losses ratio
- `rollout/ep_entries`, `ep_wins`, `ep_losses`: Episode trade statistics
- `train/approx_kl`: Should stay small (< 0.05)
- `train/entropy_loss`: Should not collapse to 0

### Tips for Better Results

1. **Start Small**: Use `--train-slice-bars 5000` to verify the model can overfit
2. **Entropy Tuning**: If buy_rate collapses to 0 or 1, adjust entropy coefficient
3. **Learning Rate**: Currently set to 1e-6 for stability; increase if learning is too slow
4. **Episode Length**: Ensure episodes are long enough for TP/SL to trigger
5. **Parallel Envs**: Use `--n-envs 8` or more for faster training (with subproc backend)

### Common Issues

- **Buy rate = 0**: Model learned not to trade. Increase entropy or add small entry bonus
- **Buy rate = 1**: Model spam buys. Check if invalid buy penalty is working
- **No TP/SL hits**: Stop loss might be too wide for the data volatility
- **High KL divergence**: Lower learning rate or tighten target_kl constraint
