# XGBoost Parameter Guide

## Understanding XGBoost vs Neural Networks

### Key Difference: No "Epochs"
- **Neural Networks**: Loop through data multiple times (epochs)
- **XGBoost**: Builds trees sequentially in ONE pass through data
- `n_estimators=200` means 200 trees built sequentially, NOT 200 epochs

### How XGBoost Learns
1. Build first tree to predict target
2. Calculate errors (residuals) from first tree
3. Build second tree to predict those errors
4. Repeat for `n_estimators` trees
5. Final prediction = sum of all tree predictions

## Core Parameters

### `--n-estimators` (Number of Trees)
**What it is:** Number of decision trees to build sequentially

**Think of it as:**
- More trees = More learning capacity (like more training steps)
- Each tree corrects errors from previous trees

**Recommendations:**
- **Fast testing**: 100-200 trees
- **Good performance**: 200-500 trees
- **Maximum performance**: 500-1000 trees (with lower learning rate)
- **With learning rate decay**: 300-500 trees works well

**Examples:**
```bash
# Quick training
--n-estimators 100

# Balanced (recommended)
--n-estimators 200 --learning-rate 0.05

# Deep learning (with low LR)
--n-estimators 500 --learning-rate 0.03 --lr-decay --lr-end 0.005
```

### `--max-depth` (Tree Complexity)
**What it is:** Maximum depth of each individual tree

**Effects:**
- **Shallow (3-5)**: Simple patterns, fast, less overfitting
- **Medium (6-8)**: Good balance, most common
- **Deep (9-12)**: Complex patterns, slow, overfitting risk

**Recommendations:**
- **Start with**: 6 (default)
- **If underfitting**: Increase to 8-10
- **If overfitting**: Decrease to 4-6

**Examples:**
```bash
# Simple patterns
--max-depth 4

# Balanced (recommended)
--max-depth 8

# Complex patterns (risk overfitting)
--max-depth 12
```

### `--learning-rate` (Step Size)
**What it is:** How much each tree contributes to the final model

**Effects:**
- **Higher (0.1-0.3)**: Fast learning, fewer trees needed, risk overfitting
- **Medium (0.05-0.1)**: Balanced
- **Lower (0.01-0.05)**: Slow learning, needs many trees, better generalization

**Recommendation:**
- Higher LR = Fewer trees needed
- Lower LR = More trees needed, but often better results

**Rule of thumb:**
```
learning_rate * n_estimators ≈ constant

Examples:
- 0.1 × 200 = 20
- 0.05 × 400 = 20
- 0.01 × 2000 = 20
```

## Recommended Configurations

### 1. Quick Training (Fast Results)
**Good for testing/iteration**
```bash
python xgboost_predictor/train_xgb.py \
  --n-estimators 100 \
  --max-depth 6 \
  --learning-rate 0.1
```
Training time: ~1-2 minutes

### 2. Balanced Performance (Recommended)
**Good general purpose**
```bash
python xgboost_predictor/train_xgb.py \
  --n-estimators 200 \
  --max-depth 8 \
  --learning-rate 0.05 \
  --lr-decay \
  --lr-end 0.01 \
  --early-stopping-rounds 20
```
Training time: ~3-5 minutes

### 3. Maximum Performance (Best Results)
**For final model**
```bash
python xgboost_predictor/train_xgb.py \
  --n-estimators 500 \
  --max-depth 8 \
  --learning-rate 0.03 \
  --lr-decay \
  --lr-end 0.005 \
  --early-stopping-rounds 50
```
Training time: ~10-15 minutes

### 4. Deep Learning Style (Most Capacity)
**Like training a neural network for many epochs**
```bash
python xgboost_predictor/train_xgb.py \
  --n-estimators 1000 \
  --max-depth 10 \
  --learning-rate 0.02 \
  --lr-decay \
  --lr-end 0.001 \
  --early-stopping-rounds 100 \
  --subsample 0.7 \
  --colsample-bytree 0.7
```
Training time: ~20-30 minutes

## Early Stopping

**What it does:** Stops training if validation metric doesn't improve

**Example:**
```bash
--early-stopping-rounds 50
```
This means: "If test AUC doesn't improve for 50 consecutive trees, stop training early"

**Benefits:**
- Prevents overfitting
- Saves time (might stop at tree 300 instead of 1000)
- Automatically finds optimal number of trees

**Recommendations:**
- Use `--early-stopping-rounds 20-50` for most cases
- Higher values (50-100) for very large `n_estimators`

## Other Parameters

### `--subsample`
**What it is:** Fraction of samples to use for each tree

- Default: 0.8 (use 80% of data per tree)
- Lower values (0.6-0.7) reduce overfitting but may underfit
- Higher values (0.9-1.0) use more data but may overfit

### `--colsample-bytree`
**What it is:** Fraction of features to use for each tree

- Default: 0.8 (use 80% of features per tree)
- Adds randomness to prevent overfitting
- Similar to dropout in neural networks

### `--min-child-weight`
**What it is:** Minimum number of samples in a leaf node

- Default: 1
- Higher values (5-10) prevent overfitting
- Too high will underfit

## How to Choose Parameters

### Start Simple
```bash
python xgboost_predictor/train_xgb.py \
  --n-estimators 200 \
  --max-depth 8 \
  --learning-rate 0.05
```

### Check Results
- Look at train vs test AUC
- **Overfitting** (train >> test): Reduce max_depth, add early stopping
- **Underfitting** (both low): Increase max_depth or n_estimators

### Optimize
- Add learning rate decay for stability
- Add early stopping to prevent overfitting
- Increase n_estimators with lower learning rate for best results

## Comparison to Your RL Training

Your RL training at 2.5B timesteps is like:

**RL (2.5B steps):**
```bash
--timesteps 2_500_000_000  # Many environment steps
--learning-rate 7e-5       # Very small steps
```

**XGBoost equivalent:**
```bash
--n-estimators 500-1000    # Many trees
--learning-rate 0.02       # Small contribution per tree
--lr-decay                 # Decay like RL
--early-stopping-rounds 50 # Prevent overtraining
```

Both approaches: **Many small steps = Better generalization**
