"""Discrete action wrapper for trading: 0=hold, 1=buy (if flat)."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DiscreteActionWrapper(gym.Wrapper):
    """Converts continuous trading env to discrete: 0=hold, 1=buy.
    
    - When flat: 0=stay flat, 1=buy 1 unit
    - When holding: any action does nothing (wait for TP/SL or other exit)
    - Observation space unchanged
    - Action space becomes Discrete(2)
    """
    
    def __init__(self, env: gym.Env, buy_amount: float = 1.0):
        super().__init__(env)
        self.buy_amount = buy_amount  # How much to buy (in continuous action units)
        # Override action space
        self.action_space = spaces.Discrete(2)
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        
    def step(self, action: int):
        # Get current position from the unwrapped base env
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        current_position = float(np.maximum(base_env.stocks, 0.0).sum())
        
        # Convert discrete to continuous
        if current_position > 0.0:
            # Already holding: do nothing
            continuous_action = np.array([0.0], dtype=np.float32)
        elif action == 0:
            # Hold action when flat
            continuous_action = np.array([0.0], dtype=np.float32)
        elif action == 1:
            # Buy action when flat
            continuous_action = np.array([self.buy_amount], dtype=np.float32)
        else:
            raise ValueError(f"Invalid discrete action: {action}")
            
        # Pass to wrapped env and add debug info
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        
        # Debug logging
        info = dict(info or {})
        info["discrete_action"] = int(action)
        info["was_flat"] = current_position <= 0.0
        if action == 1 and current_position <= 0.0:
            info["discrete_buy_attempt"] = True
            info["continuous_action_sent"] = float(continuous_action[0])
            # Check position after step
            new_position = float(np.maximum(base_env.stocks, 0.0).sum())
            info["position_after"] = new_position
            
        return obs, reward, terminated, truncated, info
