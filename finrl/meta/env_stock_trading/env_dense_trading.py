"""Dense reward trading environment for futures scalping (fixed contract size)."""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class DenseRewardTradingEnv(gym.Env):
    """Trading environment for fixed-size futures contracts (e.g., 1 MES).
    
    Actions:
        0: Hold/Do nothing
        1: Buy (if flat) or Sell (if long)
        
    Rewards are based on ABSOLUTE PnL (not % return):
        - No reward while holding (avoid mark-to-market noise)
        - Exponential reward for quality wins (big wins >> many small wins)
        - Linear penalty for losses
        - No capital/cash tracking (irrelevant for fixed-size futures)
    """
    
    def __init__(
        self,
        price_array: np.ndarray,
        tech_array: np.ndarray,
        tick_size: float = 0.25,
        contract_multiplier: float = 5.0,
        transaction_cost_bps: float = 2.0,  # ~$0.50 per side for MES
        inactivity_penalty: float = 0.0,  # Set to small value (e.g. 0.0001) if needed
        max_hold_bars: int = 30,  # Maximum bars to hold position (30 minutes for day trading)
        holding_loss_penalty: bool = True,  # Penalize holding unrealized losses
        grace_period_bars: int = 45,  # Bars before penalty kicks in (45 min grace period)
        emergency_stop_loss: float = -50.0,  # Force exit if unrealized loss exceeds this ($)
    ):
        self.price_array = price_array.astype(np.float32).flatten()
        self.tech_array = tech_array.astype(np.float32)
        self.tick_size = tick_size
        self.contract_multiplier = contract_multiplier
        self.transaction_cost_bps = transaction_cost_bps
        self.inactivity_penalty = inactivity_penalty
        self.max_hold_bars = max_hold_bars
        self.holding_loss_penalty = holding_loss_penalty
        self.grace_period_bars = grace_period_bars
        self.emergency_stop_loss = emergency_stop_loss
        
        # Spaces
        self.action_space = spaces.Discrete(2)
        # Observation: [position, bars_since_entry, price_diff_dollars, current_price, entry_price] + [rsi_14, sma_7, sma_21, atr_14]
        # Pure price action focus - position state, current & entry prices, momentum, trend, volatility
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5 + self.tech_array.shape[1],),  # 5 state + 4 tech indicators = 9 total
            dtype=np.float32
        )
        
        self.max_steps = len(self.price_array) - 1
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.position = 0  # 0 or 1
        self.entry_price = 0.0
        self.entry_day = 0
        
        # Episode stats (pure PnL tracking, no capital concept)
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.wins_no_cost = 0  # Wins excluding transaction costs
        self.losses_no_cost = 0
        self.total_pnl = 0.0  # Cumulative realized PnL in dollars
        self.total_bars_held = 0  # Track holding duration
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        price = self.price_array[self.day]
        
        # Bars held - track how long position has been held
        if self.position > 0:
            bars_since_entry = float(self.day - self.entry_day)
            price_diff = price - self.entry_price
            entry_price = self.entry_price
        else:
            bars_since_entry = 0.0
            price_diff = 0.0
            entry_price = 0.0  # 0 when flat (no entry)
        
        obs = np.array([
            float(self.position),
            bars_since_entry,
            price_diff,
            price,         # Current price (real-time, not lagged)
            entry_price,   # Entry price (0 if flat)
        ], dtype=np.float32)
        
        # Add technical indicators
        tech = self.tech_array[self.day]
        obs = np.concatenate([obs, tech])
        
        return obs
        
    def step(self, action):
        assert self.action_space.contains(action)
        
        info = {}
        price = self.price_array[self.day]
        
        # Handle actions
        if action == 1:
            if self.position == 0:
                # Buy 1 contract (open position)
                self.position = 1
                self.entry_price = price
                self.entry_day = self.day
                
                info["action"] = "buy"
                info["entry_price"] = price
                    
            elif self.position > 0:
                # Sell (close position)
                cost = price * self.contract_multiplier
                transaction_cost = cost * (self.transaction_cost_bps / 10000)
                
                # Calculate realized PnL and holding duration
                bars_held = self.day - self.entry_day
                trade_pnl = (price - self.entry_price) * self.contract_multiplier - 2 * transaction_cost
                trade_pnl_no_cost = (price - self.entry_price) * self.contract_multiplier
                
                self.total_pnl += trade_pnl
                self.total_bars_held += bars_held
                self.trades += 1
                
                # Wins/losses with transaction costs
                if trade_pnl > 0:
                    self.wins += 1
                else:
                    self.losses += 1
                
                # Wins/losses without transaction costs (to see if costs are the issue)
                if trade_pnl_no_cost > 0:
                    self.wins_no_cost += 1
                else:
                    self.losses_no_cost += 1
                
                info["action"] = "sell"
                info["exit_price"] = price
                info["trade_pnl"] = trade_pnl
                info["bars_held"] = bars_held
                info["transaction_cost"] = transaction_cost
                
                # REALIZATION REWARD: Exponential scaling for quality trades
                # Based on ABSOLUTE PnL (not relative to capital)
                # Why? Fixed contract size = absolute $ matters, not "return on capital"
                
                # Calculate total round-trip transaction cost
                total_transaction_cost = 2 * transaction_cost
                
                # Case 1: Trade was profitable before costs but NOT after costs
                # → PENALTY! Model needs to learn these trades are BAD
                if trade_pnl_no_cost > 0 and trade_pnl <= 0:
                    # Punish trades where costs ate the profit (linear)
                    realization_reward = trade_pnl / 10.0  # Small penalty
                    info["cost_eaten"] = True
                
                # Case 2: Trade is net profitable after costs
                elif trade_pnl > 0:
                    # Linear reward for profitable trades
                    realization_reward = trade_pnl / 10.0
                
                # Case 3: Trade was a loss even before costs
                else:
                    # Linear penalty for genuine losses
                    realization_reward = trade_pnl / 10.0
                
                info["realization_reward"] = realization_reward
                
                # Examples (linear scaling):
                # $2 gross, -$1 net: PENALTY -0.1 (costs ate profit!)
                # $1 net profit: reward = 0.1
                # $5 net profit: reward = 0.5
                # $25 net profit: reward = 2.5
                # $50 net profit: reward = 5.0
                # Linear scaling is more predictable for the critic
                
                # Reset position
                self.position = 0
                self.entry_price = 0.0
        
        # Advance time
        self.day += 1
        terminated = self.day >= self.max_steps

        # EMERGENCY STOP-LOSS: Force exit if loss exceeds threshold
        if self.position > 0:
            price = self.price_array[self.day]
            unrealized_pnl = (price - self.entry_price) * self.contract_multiplier

            if unrealized_pnl < self.emergency_stop_loss:
                # Force close position due to emergency stop-loss
                cost = price * self.contract_multiplier
                transaction_cost = cost * (self.transaction_cost_bps / 10000)
                trade_pnl = unrealized_pnl - 2 * transaction_cost
                trade_pnl_no_cost = unrealized_pnl

                self.total_pnl += trade_pnl
                self.total_bars_held += (self.day - self.entry_day)
                self.trades += 1

                if trade_pnl > 0:
                    self.wins += 1
                else:
                    self.losses += 1

                if trade_pnl_no_cost > 0:
                    self.wins_no_cost += 1
                else:
                    self.losses_no_cost += 1

                # EXTRA PENALTY for hitting stop-loss
                # Base PnL penalty + extra punishment for letting it get this bad
                realization_reward = trade_pnl / 10.0  # Already very negative
                realization_reward -= 5.0  # Extra -5.0 penalty for hitting emergency stop

                info["realization_reward"] = realization_reward
                info["action"] = "sell_emergency_stop"
                info["exit_price"] = price
                info["trade_pnl"] = trade_pnl
                info["bars_held"] = self.day - self.entry_day
                info["emergency_stop"] = True

                # Reset position
                self.position = 0
                self.entry_price = 0.0

        # Check for max hold time (force exit if held too long)
        if self.position > 0 and (self.day - self.entry_day) >= self.max_hold_bars:
            # Force close position due to max hold time
            cost = price * self.contract_multiplier
            transaction_cost = cost * (self.transaction_cost_bps / 10000)
            trade_pnl = (price - self.entry_price) * self.contract_multiplier - 2 * transaction_cost
            trade_pnl_no_cost = (price - self.entry_price) * self.contract_multiplier
            
            self.total_pnl += trade_pnl
            self.total_bars_held += (self.day - self.entry_day)
            self.trades += 1
            
            if trade_pnl > 0:
                self.wins += 1
            else:
                self.losses += 1
                
            if trade_pnl_no_cost > 0:
                self.wins_no_cost += 1
            else:
                self.losses_no_cost += 1
            
            # Apply same reward logic as manual exit
            if trade_pnl_no_cost > 0 and trade_pnl <= 0:
                realization_reward = trade_pnl / 10.0
                info["cost_eaten"] = True
            elif trade_pnl > 0:
                realization_reward = trade_pnl / 10.0
            else:
                realization_reward = trade_pnl / 10.0
            
            info["realization_reward"] = realization_reward
            info["action"] = "sell_forced"
            info["exit_price"] = price
            info["trade_pnl"] = trade_pnl
            info["bars_held"] = self.day - self.entry_day
            info["forced_exit"] = True
            
            # Reset position
            self.position = 0
            self.entry_price = 0.0
        
        # Reward structure (sparse):
        # - While holding: reward = 0 (no unrealized PnL)
        # - On realization: reward = PnL only (transaction costs already in PnL calc)
        reward = 0.0

        # Pure sparse reward: only on realization (exit)
        if "realization_reward" in info:
            reward = info["realization_reward"]

        # Graduated penalty for holding unrealized losses
        # Gives trades time to work out, but penalizes holding big losers
        if self.holding_loss_penalty and self.position > 0:
            unrealized_pnl = (price - self.entry_price) * self.contract_multiplier
            bars_held = self.day - self.entry_day

            # Only penalize losses after grace period
            if unrealized_pnl < 0 and bars_held > self.grace_period_bars:
                # Penalty grows with time AND loss size
                # time_factor: ramps from 0 to 1 as we approach max_hold_bars
                time_over_grace = bars_held - self.grace_period_bars
                time_until_forced = self.max_hold_bars - self.grace_period_bars
                time_factor = min(time_over_grace / time_until_forced, 1.0)

                # Apply graduated penalty: small at first, stronger near forced exit
                # Scale: unrealized_pnl is negative, so this adds a negative reward
                holding_penalty = (unrealized_pnl / 200.0) * time_factor
                reward += holding_penalty

                info["holding_penalty"] = holding_penalty
                info["time_factor"] = time_factor

        # Optional: small penalty for inactivity
        if self.inactivity_penalty > 0 and self.position == 0:
            reward -= self.inactivity_penalty
        
        # Force close position at end
        if terminated and self.position > 0:
            cost = price * self.contract_multiplier
            transaction_cost = cost * (self.transaction_cost_bps / 10000)
            trade_pnl = (price - self.entry_price) * self.contract_multiplier - 2 * transaction_cost
            trade_pnl_no_cost = (price - self.entry_price) * self.contract_multiplier
            
            self.total_pnl += trade_pnl
            self.total_bars_held += (self.day - self.entry_day)
            self.trades += 1
            
            if trade_pnl > 0:
                self.wins += 1
            else:
                self.losses += 1
                
            if trade_pnl_no_cost > 0:
                self.wins_no_cost += 1
            else:
                self.losses_no_cost += 1
                
            self.position = 0
        
        # Episode summary on termination
        if terminated:
            info["ep_trades"] = self.trades
            info["ep_wins"] = self.wins
            info["ep_losses"] = self.losses
            info["ep_wins_no_cost"] = self.wins_no_cost
            info["ep_losses_no_cost"] = self.losses_no_cost
            info["ep_total_pnl"] = self.total_pnl
            info["ep_avg_bars_held"] = self.total_bars_held / self.trades if self.trades > 0 else 0
            # No "return %" - just absolute PnL matters for fixed-size futures
        
        obs = self._get_obs()
        return obs, reward, terminated, False, info
