#!/bin/bash
# Setup script for vast.ai
# Run this after uploading your code to vast.ai

set -e  # Exit on error

echo "=== Setting up MES Predictor on vast.ai ==="

# Update system
echo "Updating system packages..."
apt-get update -qq
apt-get install -y -qq wget build-essential

# Install TA-Lib (optional but recommended)
echo "Installing TA-Lib..."
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -q
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make -j$(nproc)
make install
ldconfig
cd ~

# Install Python dependencies
echo "Installing Python packages..."
cd ~/mes-predictor  # Adjust path if needed
pip install --upgrade pip -q
pip install -r requirements_vastai.txt -q

# Verify installation
echo "Verifying installation..."
python -c "import stable_baselines3; print('SB3:', stable_baselines3.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

echo "=== Setup complete! ==="
echo ""
echo "To start training:"
echo "  cd ~/mes-predictor"
echo "  python scripts/train_dense.py --name ppo_vastai --timesteps 100000000 --train-slice 350000 --n-envs 32 --transaction-cost 1.0"
