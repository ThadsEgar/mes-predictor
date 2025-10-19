from __future__ import annotations

from typing import Any, Dict, List
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeStatsCallback(BaseCallback):
    """Logs per-episode and per-event trade stats to TensorBoard.

    Expects infos possibly containing:
      - ep_trades, ep_wins, ep_losses, ep_realized_pnl, ep_entries
      - entry (bool), entry_price, entry_notional_usd
      - tp_sl_exit ("stop"/"target"), trade_realized_pnl, *_usd variants
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log action distribution for discrete actions
        if "actions" in self.locals:
            actions = self.locals["actions"]
            # For discrete, log the percentage of buy actions (action=1)
            buy_rate = float((actions == 1).mean())
            self.logger.record("rollout/buy_rate", buy_rate)
        
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])
        if not infos:
            return True
        for info in infos:
            if not isinstance(info, dict):
                continue
            # Episode-level metrics (logged when episode ends)
            if "ep_trades" in info:
                self.logger.record("rollout/ep_trades", int(info.get("ep_trades", 0)))
            if "ep_wins" in info:
                self.logger.record("rollout/ep_wins", int(info.get("ep_wins", 0)))
            if "ep_losses" in info:
                self.logger.record("rollout/ep_losses", int(info.get("ep_losses", 0)))
            if "ep_entries" in info:
                self.logger.record("rollout/ep_entries", int(info.get("ep_entries", 0)))
            if "ep_realized_pnl" in info:
                self.logger.record("rollout/ep_realized_pnl", float(info.get("ep_realized_pnl", 0.0)))
            if "ep_realized_pnl_usd" in info:
                self.logger.record("rollout/ep_realized_pnl_usd", float(info.get("ep_realized_pnl_usd", 0.0)))

            # Derived episode metrics
            if "ep_wins" in info or "ep_losses" in info:
                wins = int(info.get("ep_wins", 0))
                losses = int(info.get("ep_losses", 0))
                trades = wins + losses
                if trades > 0:
                    win_rate = wins / trades
                    self.logger.record("rollout/win_rate", float(win_rate))
                if losses > 0:
                    wl_ratio = wins / losses
                    self.logger.record("rollout/win_loss_ratio", float(wl_ratio))

            # Per-event logs (sparse): entries and exits
            if info.get("entry"):
                self.logger.record("rollout/entry_event", 1)
                if "entry_notional_usd" in info:
                    self.logger.record("rollout/entry_notional_usd", float(info.get("entry_notional_usd", 0.0)))

            if "trade_realized_pnl" in info:
                self.logger.record("rollout/trade_realized_pnl", float(info.get("trade_realized_pnl", 0.0)))
            if "trade_realized_pnl_usd" in info:
                self.logger.record("rollout/trade_realized_pnl_usd", float(info.get("trade_realized_pnl_usd", 0.0)))
            if "tp_sl_exit" in info:
                exit_type = info.get("tp_sl_exit")
                # Encode target=1, stop=0 for a quick signal
                code = 1 if exit_type == "target" else (0 if exit_type == "stop" else -1)
                self.logger.record("rollout/tp_sl_exit_code", code)
                # Separate hit counters (sparse increments)
                if exit_type == "target":
                    self.logger.record("rollout/target_hit", 1)
                elif exit_type == "stop":
                    self.logger.record("rollout/stop_hit", 1)
            
        return True
