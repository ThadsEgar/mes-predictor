"""Callbacks for logging metrics from dense reward trading environment."""

from typing import Any, Dict, List
from stable_baselines3.common.callbacks import BaseCallback


class DenseEnvStatsCallback(BaseCallback):
    """Logs per-episode and per-action stats for dense reward environment.
    
    Tracks:
    - Buy/sell actions per rollout
    - Episode trades, wins, losses, PnL
    - Portfolio return percentage
    - Trade quality flags (cost_eaten)
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.buy_count = 0
        self.sell_count = 0
        self.rollout_steps = 0
        
    def _on_step(self) -> bool:
        # Count actions in current rollout
        if "actions" in self.locals:
            actions = self.locals["actions"]
            self.rollout_steps += len(actions)
            
            # For discrete actions, track buy attempts (action=1)
            buy_actions = (actions == 1).sum()
            self.buy_count += buy_actions
            
        # Process info dicts for trades and episode stats
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
                
            # Track individual trades
            if "action" in info:
                if info["action"] == "sell":
                    self.sell_count += 1
                    # Log PnL and holding duration of each trade
                    if "trade_pnl" in info:
                        self.logger.record("rollout/trade_pnl", float(info["trade_pnl"]))
                    if "bars_held" in info:
                        self.logger.record("rollout/bars_held", int(info["bars_held"]))
                    # Log trade quality flags
                    if info.get("cost_eaten", False):
                        self.logger.record("rollout/cost_eaten_trade", 1.0)
                        
            # Episode-level stats
            if "ep_trades" in info:
                self.logger.record("rollout/ep_trades", int(info["ep_trades"]))
                self.logger.record("rollout/ep_wins", int(info.get("ep_wins", 0)))
                self.logger.record("rollout/ep_losses", int(info.get("ep_losses", 0)))
                
                wins = int(info.get("ep_wins", 0))
                losses = int(info.get("ep_losses", 0))
                if wins + losses > 0:
                    win_rate = wins / (wins + losses)
                    self.logger.record("rollout/win_rate", win_rate)
                    
                # Track wins/losses without transaction costs
                wins_no_cost = int(info.get("ep_wins_no_cost", 0))
                losses_no_cost = int(info.get("ep_losses_no_cost", 0))
                if wins_no_cost + losses_no_cost > 0:
                    win_rate_no_cost = wins_no_cost / (wins_no_cost + losses_no_cost)
                    self.logger.record("rollout/win_rate_no_cost", win_rate_no_cost)
                    self.logger.record("rollout/ep_wins_no_cost", wins_no_cost)
                    self.logger.record("rollout/ep_losses_no_cost", losses_no_cost)
                    
            if "ep_avg_bars_held" in info:
                self.logger.record("rollout/ep_avg_bars_held", float(info["ep_avg_bars_held"]))
                    
            if "ep_total_pnl" in info:
                self.logger.record("rollout/ep_total_pnl", float(info["ep_total_pnl"]))
                
        return True
        
    def _on_rollout_end(self) -> None:
        # Log action rates for the rollout
        if self.rollout_steps > 0:
            buy_rate = self.buy_count / self.rollout_steps
            self.logger.record("rollout/buy_rate", buy_rate)
            
            if self.sell_count > 0:
                buy_sell_ratio = self.buy_count / self.sell_count if self.sell_count > 0 else 0
                self.logger.record("rollout/buy_sell_ratio", buy_sell_ratio)
                
        # Reset counters for next rollout
        self.buy_count = 0
        self.sell_count = 0
        self.rollout_steps = 0
