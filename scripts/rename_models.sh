#!/bin/bash
# Rename RL models to sequential naming scheme

cd models

echo "Renaming models to sequential scheme..."
echo "=========================================="

# Keep the old names but create copies with new sequential names
# Based on chronological order and total training steps

echo "Model 1: 1B steps, 1.0 bps cost"
if [ -f "ppo_2_years_1B_1.0_transaction_cost-graduated" ]; then
    cp -p "ppo_2_years_1B_1.0_transaction_cost-graduated" "rl_model_v1_1B_1.0bps"
    cp -p "ppo_2_years_1B_1.0_transaction_cost-graduated_vecnormalize.pkl" "rl_model_v1_1B_1.0bps_vecnormalize.pkl"
    echo "  ✓ Created rl_model_v1_1B_1.0bps"
fi

echo "Model 2: 1B steps, 0.0 bps cost"
if [ -f "ppo_2_years_1B_0.0_transaction_cost-graduated" ]; then
    cp -p "ppo_2_years_1B_0.0_transaction_cost-graduated" "rl_model_v2_1B_0.0bps"
    cp -p "ppo_2_years_1B_0.0_transaction_cost-graduated_vecnormalize.pkl" "rl_model_v2_1B_0.0bps_vecnormalize.pkl"
    echo "  ✓ Created rl_model_v2_1B_0.0bps"
fi

echo "Model 3: 1.5B steps, 0.5→1.0 bps curriculum"
if [ -f "ppo_2_years_1.5B_0.5to1.0_cost_curriculum" ]; then
    cp -p "ppo_2_years_1.5B_0.5to1.0_cost_curriculum" "rl_model_v3_1.5B_curriculum"
    cp -p "ppo_2_years_1.5B_0.5to1.0_cost_curriculum_vecnormalize.pkl" "rl_model_v3_1.5B_curriculum_vecnormalize.pkl"
    echo "  ✓ Created rl_model_v3_1.5B_curriculum"
fi

echo "Model 4: 2B steps, 0.5 bps fixed"
if [ -f "ppo_2_years_2B_0.5_fixed_cost" ]; then
    cp -p "ppo_2_years_2B_0.5_fixed_cost" "rl_model_v4_2B_0.5bps"
    cp -p "ppo_2_years_2B_0.5_fixed_cost_vecnormalize.pkl" "rl_model_v4_2B_0.5bps_vecnormalize.pkl"
    echo "  ✓ Created rl_model_v4_2B_0.5bps"
fi

echo "Model 5: 2B steps, 0.5 bps fixed (iteration 1)"
if [ -f "ppo_2_years_2B_0.5_fixed_cost_1" ]; then
    cp -p "ppo_2_years_2B_0.5_fixed_cost_1" "rl_model_v5_2B_0.5bps"
    cp -p "ppo_2_years_2B_0.5_fixed_cost_1_vecnormalize.pkl" "rl_model_v5_2B_0.5bps_vecnormalize.pkl"
    echo "  ✓ Created rl_model_v5_2B_0.5bps"
fi

echo "Model 6: 2.5B steps, 0.5 bps fixed"
if [ -f "ppo_2_years_2.5B_0.5_fixed_cost_2" ]; then
    cp -p "ppo_2_years_2.5B_0.5_fixed_cost_2" "rl_model_v6_2.5B_0.5bps"
    cp -p "ppo_2_years_2.5B_0.5_fixed_cost_2_vecnormalize.pkl" "rl_model_v6_2.5B_0.5bps_vecnormalize.pkl"
    echo "  ✓ Created rl_model_v6_2.5B_0.5bps"
fi

echo "Model 7: 3B steps, 0.5 bps fixed (LATEST)"
if [ -f "ppo_2_years_3B_0.5_fixed_cost_3" ]; then
    cp -p "ppo_2_years_3B_0.5_fixed_cost_3" "rl_model_v7_3B_0.5bps_LATEST"
    cp -p "ppo_2_years_3B_0.5_fixed_cost_3_vecnormalize.pkl" "rl_model_v7_3B_0.5bps_LATEST_vecnormalize.pkl"
    echo "  ✓ Created rl_model_v7_3B_0.5bps_LATEST"
fi

echo ""
echo "=========================================="
echo "✓ Done! Sequential models created."
echo ""
echo "Your latest model:"
echo "  rl_model_v7_3B_0.5bps_LATEST"
echo ""
echo "Old models are preserved with original names."
echo "New sequential names are easier to track!"
