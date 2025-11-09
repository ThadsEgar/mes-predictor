# XGBoost 1:2 Risk-Reward Predictor

This folder contains an XGBoost-based approach to predict whether a 1:2 risk-reward ratio trade is achievable at any given timestep.

## Overview

Unlike the reinforcement learning approach, this uses supervised learning to predict binary outcomes:
- **Label 1**: Target will be hit before stop loss (1:2 RR achieved)
- **Label 0**: Stop loss will be hit first (trade fails)

**📖 See [PARAMETERS.md](PARAMETERS.md) for detailed guide on XGBoost parameters and recommended configurations**

## Features Used

The model uses the same technical indicators as the RL approach:
- RSI(14)
- SMA(7) and SMA(21)
- ATR(14)
- Price relationships (close to SMAs, SMA crossovers)
- Volatility features (high-low range, close-open)
- Momentum features (price changes over 1, 5, 20 bars)

## Workflow

### 1. Prepare Labeled Data

Label the data by looking forward to see if 1:2 RR is achievable:

```bash
python xgboost_predictor/prepare_data.py \
  --csv datasets/mes_finrl_ready_front.csv \
  --output xgboost_predictor/labeled_data.csv \
  --risk-dollars 5.0 \
  --train-slice 350000
```

**Arguments:**
- `--risk-dollars`: Risk per trade in dollars (default: 5.0)
- `--contract-multiplier`: Contract multiplier (default: 5.0 for MES)
- `--lookahead-bars`: Max bars to look ahead (default: 240 = 4 hours)
- `--train-slice`: Use only last N bars (optional)

### 2. Hyperparameter Sweep (Recommended)

Automatically find the best parameters:

```bash
# Random search (fast, recommended)
python xgboost_predictor/sweep_params.py \
  --search-type random \
  --n-iter 50 \
  --train-best

# Grid search (exhaustive, slower)
python xgboost_predictor/sweep_params.py \
  --search-type grid \
  --train-best
```

**Arguments:**
- `--search-type`: `random` (faster) or `grid` (exhaustive)
- `--n-iter`: Number of random combinations to try (default: 50)
- `--metric`: Optimize for `roc_auc`, `f1`, `precision`, or `recall`
- `--train-best`: Automatically train final model with best params

**Outputs:**
- `sweep_results/sweep_*.csv`: All parameter combinations tested
- `sweep_results/best_params_*.json`: Best parameters found
- `models/xgb_1_2_rr_best.pkl`: Trained model (if --train-best used)

### 3. Train XGBoost Model Manually

Or train with specific parameters:

```bash
python xgboost_predictor/train_xgb.py \
  --time-series-split \
  --n-estimators 200 \
  --max-depth 8 \
  --learning-rate 0.05 \
  --early-stopping-rounds 50
```

**Arguments:**
- `--n-estimators`: Number of boosting rounds (default: 100)
- `--max-depth`: Max tree depth (default: 6)
- `--learning-rate`: Starting learning rate (default: 0.1)
- `--lr-decay`: Enable learning rate decay
- `--lr-end`: Final learning rate for decay (default: 0.01)
- `--test-size`: Test set fraction (default: 0.2)
- `--scale-pos-weight`: Balance positive/negative classes (auto if None)

**Outputs:**
- `models/xgb_1_2_rr.pkl`: Trained model
- `models/feature_importance.png`: Feature importance plot
- `models/confusion_matrix.png`: Confusion matrix
- `models/precision_recall_curve.png`: Precision-recall curve
- `models/optimal_threshold.txt`: Optimal probability threshold

### 4. Evaluate Model

Simulate trading with the model (use `--test-only` to avoid overfitting!):

```bash
# Evaluate on test set only (realistic performance)
python xgboost_predictor/evaluate_xgb.py \
  --model xgboost_predictor/models/xgb_1_2_rr_best.pkl \
  --test-only \
  --transaction-cost 0.5
```

**Arguments:**
- `--test-only`: Evaluate only on unseen test data (IMPORTANT!)
- `--threshold`: Prediction probability threshold (default: load from optimal_threshold.txt)
- `--transaction-cost`: Transaction cost in bps (default: 0.5)
- `--test-size`: Test set size (default: 0.2 = 20%)

**Outputs:**
- `eval_results/threshold_comparison_*.csv`: Performance at different thresholds
- `eval_results/predictions_*.csv`: Detailed predictions for each timestep

## Example: Full Pipeline

```bash
# 1. Prepare data (350k bars, $5 risk)
python xgboost_predictor/prepare_data.py \
  --train-slice 350000 \
  --risk-dollars 5.0

# 2. Hyperparameter sweep (finds best params automatically)
python xgboost_predictor/sweep_params.py \
  --search-type random \
  --n-iter 50 \
  --train-best

# 3. Evaluate on test set only (realistic performance)
python xgboost_predictor/evaluate_xgb.py \
  --model xgboost_predictor/models/xgb_1_2_rr_best.pkl \
  --test-only \
  --transaction-cost 0.5
```

## Interpretation

The model outputs probabilities (0 to 1) that indicate confidence:
- **High probability (>0.7)**: Model is confident target will be hit
- **Low probability (<0.3)**: Model predicts stop will be hit first
- **Medium probability (0.3-0.7)**: Uncertain outcome

You can adjust the threshold to trade-off between:
- **Higher threshold**: Fewer trades, higher precision, potentially higher win rate
- **Lower threshold**: More trades, lower precision, more opportunities

## Comparison to RL Approach

**XGBoost (this folder):**
- ✅ Simpler, faster to train
- ✅ Interpretable (feature importance)
- ✅ Clear binary decision
- ❌ Fixed stop/target distances
- ❌ Doesn't learn entry timing dynamically

**RL (main scripts/):**
- ✅ Learns optimal entry/exit timing
- ✅ Adapts stop/target dynamically
- ✅ Considers multi-step scenarios
- ❌ Slower to train
- ❌ Less interpretable
