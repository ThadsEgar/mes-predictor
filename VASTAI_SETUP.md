# Vast.ai Setup Guide

## 1. Prepare Your Code Locally

```bash
# Create a tarball with everything needed
cd /Users/michael/Code/thads_projects/
tar -czf mes-predictor.tar.gz \
  mes-predictor/scripts/ \
  mes-predictor/finrl/ \
  mes-predictor/datasets/mes_finrl_ready_front.csv \
  mes-predictor/requirements_vastai.txt \
  mes-predictor/setup_vastai.sh \
  mes-predictor/models/ppo_2_years_500M_1.0_transaction_cost_more_ticks.zip \
  mes-predictor/models/ppo_2_years_500M_1.0_transaction_cost_more_ticks_vecnormalize.pkl
```

## 2. Rent a GPU on vast.ai

1. Go to https://vast.ai/console/create/
2. **Recommended specs:**
   - GPU: RTX 3090 or better (24GB VRAM)
   - Disk: 50GB+ SSD
   - RAM: 32GB+
   - Image: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel`
3. Click "Rent" and wait for instance to start

## 3. Upload Your Code

### Option A: Direct Upload (easiest)
```bash
# After instance starts, you'll get SSH info like:
# ssh -p 12345 root@ssh.vast.ai -L 8080:localhost:8080

# Upload tarball
scp -P 12345 mes-predictor.tar.gz root@ssh.vast.ai:~/

# SSH into instance
ssh -p 12345 root@ssh.vast.ai -L 8080:localhost:8080
```

### Option B: Git (if you have a private repo)
```bash
ssh -p 12345 root@ssh.vast.ai
git clone <your-repo-url>
```

## 4. Setup on vast.ai

```bash
# Extract code
cd ~
tar -xzf mes-predictor.tar.gz

# Run setup script
cd mes-predictor
chmod +x setup_vastai.sh
./setup_vastai.sh
```

## 5. Start Training

```bash
# Continue training from your existing model
python scripts/train_dense.py \
  --load-model models/ppo_2_years_500M_1.0_transaction_cost_more_ticks \
  --name ppo_vastai_continued \
  --transaction-cost 1.0 \
  --timesteps 200000000 \
  --train-slice 350000 \
  --n-envs 32 \
  --learning-rate 3e-6 \
  --lr-decay \
  --lr-end 1e-6 \
  --ent-coef 0.05 \
  --ent-decay \
  --ent-end 0.01

# Or start fresh training
python scripts/train_dense.py \
  --name ppo_vastai_fresh \
  --transaction-cost 1.0 \
  --timesteps 200000000 \
  --train-slice 350000 \
  --n-envs 32
```

## 6. Monitor Training

### View TensorBoard (in browser)
```bash
# On vast.ai instance:
tensorboard --logdir=runs/tensorboard --host 0.0.0.0 --port 8080

# On your local machine, open:
# http://localhost:8080
```

### Check Progress
```bash
# Watch training output
tail -f nohup.out  # if running in background

# Check GPU usage
nvidia-smi

# Check models
ls -lh models/
```

## 7. Download Results

```bash
# From your local machine:
scp -P 12345 root@ssh.vast.ai:~/mes-predictor/models/ppo_vastai_*.zip ~/Downloads/
scp -P 12345 root@ssh.vast.ai:~/mes-predictor/models/ppo_vastai_*_vecnormalize.pkl ~/Downloads/
```

## 8. Run in Background (tmux)

```bash
# Start tmux session
tmux new -s training

# Start training
python scripts/train_dense.py ...

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t training
```

## Tips

- **Cost:** RTX 3090 costs ~$0.20-0.40/hour
- **Training time:** 200M steps with 32 envs ≈ 6-12 hours
- **Save often:** Model saves automatically, but check periodically
- **Stop instance:** When done, destroy instance to stop charges

## Troubleshooting

### CUDA out of memory
```bash
# Reduce parallel envs
--n-envs 16  # instead of 32
```

### Slow training
```bash
# Check if using GPU
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU
watch -n 1 nvidia-smi
```

### Lost connection
```bash
# Training continues in tmux
# Just reconnect and: tmux attach -t training
```
