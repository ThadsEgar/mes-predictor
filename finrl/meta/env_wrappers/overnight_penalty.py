from __future__ import annotations

import pandas as pd
import numpy as np
import gymnasium as gym


class TradingRulesWrapper(gym.Wrapper):
    """
    Trading rules: penalties, position cap, entry bonus, TP/SL exits, per-episode stats.
    Options to make the agent only time entries and get sparse rewards only at TP/SL.
    """

    def __init__(
        self,
        env: gym.Env,
        time_index: pd.Series | pd.Index,
        penalty_bps: float = 0.0,
        boundary_tz: str = "America/New_York",
        boundary_hour: int = 17,
        contract_multiplier: float = 1.0,
        position_cap_contracts: float | int | None = None,
        entry_bonus_bps: float = 0.0,
        entry_bonus_min_shares: float = 1.0,
        entry_bonus_cooldown_steps: int = 0,
        entry_bonus_max_count: int = 0,
        stop_ticks: int = 0,
        rr: float = 2.0,
        slippage_bps: float = 0.0,
        fee_per_contract: float = 0.0,
        precedence_stop_first: bool = True,
        trailing_on: bool = False,
        trailing_trigger_ticks: int | None = None,
        trailing_offset_ticks: int = 0,
        log_multiplier: float = 5.0,  # 5 USD/point for MES
        # New entry/exit policy & reward controls
        entry_only_mode: bool = True,
        use_ohlc_hits: bool = False,  # current env lacks OHLC; falls back to close
        exit_reward_mode: str = "pnl",  # 'pnl' or 'binary'
        step_mark_to_market: bool = False,
        time_penalty_per_bar: float = 0.0,
    ):
        super().__init__(env)
        # Penalty / cap
        self.penalty_bps = float(penalty_bps)
        self.boundary_tz = boundary_tz
        self.boundary_hour = int(boundary_hour)
        self.contract_multiplier = float(contract_multiplier)
        self.position_cap = None if position_cap_contracts is None else float(position_cap_contracts)
        # Entry bonus
        self.entry_bonus_bps = float(entry_bonus_bps)
        self.entry_bonus_min_shares = float(entry_bonus_min_shares)
        self.entry_bonus_cooldown_steps = int(entry_bonus_cooldown_steps)
        self.entry_bonus_max_count = int(entry_bonus_max_count)
        # TP/SL
        self.stop_ticks = int(stop_ticks)
        self.rr = float(rr)
        self.slippage_bps = float(slippage_bps)
        self.fee_per_contract = float(fee_per_contract)
        self.precedence_stop_first = bool(precedence_stop_first)
        self.trailing_on = bool(trailing_on)
        self.trailing_trigger_ticks = trailing_trigger_ticks
        self.trailing_offset_ticks = int(trailing_offset_ticks)
        self.tick_size = 0.25  # MES
        # Logging conversion
        self.log_multiplier = float(log_multiplier)
        # Entry/exit policy & reward
        self.entry_only_mode = bool(entry_only_mode)
        self.use_ohlc_hits = bool(use_ohlc_hits)
        self.exit_reward_mode = exit_reward_mode
        self.step_mark_to_market = bool(step_mark_to_market)
        self.time_penalty_per_bar = float(time_penalty_per_bar)
        # Time index
        ts = pd.to_datetime(pd.Series(time_index))
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(self.boundary_tz)
        else:
            ts = ts.dt.tz_convert(self.boundary_tz)
        self._ts_local = ts
        # Runtime state
        self._prev_idx: int | None = None
        self._last_entry_idx: int = -10**9
        self._entry_bonus_count: int = 0
        self._entry_price: float | None = None
        self._stop_price: float | None = None
        self._target_price: float | None = None
        # Episode stats (index units)
        self._ep_trades: int = 0
        self._ep_wins: int = 0
        self._ep_losses: int = 0
        self._ep_realized_pnl: float = 0.0
        self._ep_entries: int = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_idx = 0
        self._last_entry_idx = -10**9
        self._entry_bonus_count = 0
        self._entry_price = None
        self._stop_price = None
        self._target_price = None
        self._ep_trades = 0
        self._ep_wins = 0
        self._ep_losses = 0
        self._ep_realized_pnl = 0.0
        self._ep_entries = 0
        return obs, info

    def _session_date(self, idx: int) -> pd.Timestamp.date:
        t = self._ts_local.iloc[idx]
        return (t - pd.Timedelta(hours=self.boundary_hour)).date()

    def _apply_overnight_penalty(self, prev_idx: int, curr_idx: int) -> tuple[float, float]:
        if self.penalty_bps <= 0.0:
            return 0.0, 0.0
        try:
            if curr_idx < len(self._ts_local) and prev_idx < len(self._ts_local):
                if self._session_date(prev_idx) != self._session_date(curr_idx):
                    price = self.env.price_ary[curr_idx]
                    notional = float((self.env.stocks * price).sum() * self.contract_multiplier)
                    if notional > 0:
                        penalty = notional * (self.penalty_bps / 1e4)
                        scaled = penalty * getattr(self.env, "reward_scaling", 1.0)
                        return penalty, scaled
        except Exception:
            return 0.0, 0.0
        return 0.0, 0.0

    def _enforce_position_cap(self, curr_idx: int) -> float:
        if self.position_cap is None:
            return 0.0
        try:
            excess = np.maximum(self.env.stocks - self.position_cap, 0.0)
            if excess.sum() <= 0:
                return 0.0
            price = self.env.price_ary[curr_idx]
            proceeds = float((price * excess).sum() * (1 - getattr(self.env, "sell_cost_pct", 0.0)))
            self.env.stocks = self.env.stocks - excess
            self.env.stocks = np.maximum(self.env.stocks, 0.0)
            self.env.amount += proceeds
            self.env.total_asset = float(self.env.amount + (self.env.stocks * price).sum())
            return proceeds
        except Exception:
            return 0.0

    def _apply_entry_bonus(self, prev_stocks: np.ndarray, curr_idx: int) -> tuple[float, float]:
        if self.entry_bonus_bps <= 0.0:
            return 0.0, 0.0
        try:
            prev_expo = float(np.maximum(prev_stocks, 0.0).sum())
            new_expo = float(np.maximum(self.env.stocks, 0.0).sum())
            executed = float((np.maximum(self.env.stocks, 0.0) - np.maximum(prev_stocks, 0.0)).sum())
            if prev_expo <= 0.0 and new_expo > 0.0 and executed >= self.entry_bonus_min_shares:
                if (self.env.day - self._last_entry_idx) < self.entry_bonus_cooldown_steps:
                    return 0.0, 0.0
                if self.entry_bonus_max_count > 0 and self._entry_bonus_count >= self.entry_bonus_max_count:
                    return 0.0, 0.0
                price = self.env.price_ary[curr_idx]
                notional = float((self.env.stocks * price).sum() * self.contract_multiplier)
                if notional <= 0:
                    return 0.0, 0.0
                bonus = notional * (self.entry_bonus_bps / 1e4)
                scaled = bonus * getattr(self.env, "reward_scaling", 1.0)
                self._entry_bonus_count += 1
                self._last_entry_idx = self.env.day
                return bonus, scaled
        except Exception:
            return 0.0, 0.0
        return 0.0, 0.0

    def _maybe_set_tp_sl_on_entry(self, prev_stocks: np.ndarray, curr_idx: int) -> None:
        if self.stop_ticks <= 0:
            return
        prev_expo = float(np.maximum(prev_stocks, 0.0).sum())
        new_expo = float(np.maximum(self.env.stocks, 0.0).sum())
        if prev_expo <= 0.0 and new_expo > 0.0:
            price = float(self.env.price_ary[curr_idx].mean() if np.ndim(self.env.price_ary[curr_idx]) > 0 else self.env.price_ary[curr_idx])
            self._entry_price = price
            stop_dist = self.stop_ticks * self.tick_size
            target_dist = self.rr * self.stop_ticks * self.tick_size
            self._stop_price = price - stop_dist
            self._target_price = price + target_dist
            if self.trailing_on and (self.trailing_trigger_ticks is None):
                self.trailing_trigger_ticks = int(self.rr * self.stop_ticks // 2)

    def _maybe_trail_stop(self, curr_idx: int) -> None:
        if not self.trailing_on or self._entry_price is None or self._stop_price is None:
            return
        price = float(self.env.price_ary[curr_idx].mean() if np.ndim(self.env.price_ary[curr_idx]) > 0 else self.env.price_ary[curr_idx])
        trigger_dist = (self.trailing_trigger_ticks or 0) * self.tick_size
        if trigger_dist <= 0:
            return
        if (price - self._entry_price) >= trigger_dist:
            new_stop = self._entry_price + self.trailing_offset_ticks * self.tick_size
            self._stop_price = max(self._stop_price, new_stop)

    def _maybe_exit_tp_sl(self, curr_idx: int) -> tuple[bool, float, str | None, float]:
        """Return (exited, additional_reward_scaled, exit_type, realized_pnl_index_units)."""
        if self._entry_price is None or self._stop_price is None or self._target_price is None:
            return False, 0.0, None, 0.0
        # OHLC not available: fall back to close; could be extended if env exposes OHLC
        price = float(self.env.price_ary[curr_idx].mean() if np.ndim(self.env.price_ary[curr_idx]) > 0 else self.env.price_ary[curr_idx])
        exit_hit = None
        if price <= self._stop_price:
            exit_hit = "stop"
        elif price >= self._target_price:
            exit_hit = "target"
        if exit_hit is None:
            return False, 0.0, None, 0.0
        holdings = np.maximum(self.env.stocks, 0.0)
        contracts = float(holdings.sum())
        if contracts <= 0:
            self._entry_price = self._stop_price = self._target_price = None
            return False, 0.0, None, 0.0
        pre_asset = float(self.env.total_asset)
        sell_price = price
        proceeds = sell_price * contracts * (1 - getattr(self.env, "sell_cost_pct", 0.0))
        notional = sell_price * contracts * self.contract_multiplier
        slip_cost = notional * (self.slippage_bps / 1e4)
        fee_cost = self.fee_per_contract * contracts
        proceeds_cash = proceeds * self.contract_multiplier - slip_cost - fee_cost
        self.env.amount += proceeds_cash
        self.env.stocks[:] = 0
        self.env.total_asset = float(self.env.amount + (self.env.stocks * sell_price).sum())
        delta_asset = self.env.total_asset - pre_asset  # index units if contract_multiplier=1
        # Reward on exit only
        if self.exit_reward_mode == "binary":
            base_scaled = (1.0 if exit_hit == "target" else -1.0)
            scaled = base_scaled  # no reward_scaling to keep +/-1
        else:
            scaled = delta_asset * getattr(self.env, "reward_scaling", 1.0)
        # episode stats
        self._ep_trades += 1
        if exit_hit == "target":
            self._ep_wins += 1
        else:
            self._ep_losses += 1
        self._ep_realized_pnl += float(delta_asset)
        # clear entry
        self._entry_price = self._stop_price = self._target_price = None
        return True, scaled, exit_hit, float(delta_asset)

    def step(self, action):
        # Ensure action is a numpy array
        action_arr = np.asarray(action, dtype=float)
        
        # Get current position
        current_position = float(np.maximum(self.env.stocks, 0.0).sum())
        
        # Entry-only mode logic
        attempting_entry = False
        if self.entry_only_mode:
            if current_position > 0.0:
                # Already holding: ignore action, wait for TP/SL
                action_arr[...] = 0.0
            elif np.any(action_arr > 0):
                # Flat and model wants to buy: force to minimal buy action
                # This ensures we buy exactly 1 contract with position cap
                min_buy_action = 0.51  # Just over 0.5 so int(0.51 * 2) = 1 contract
                action_arr[...] = min_buy_action
                attempting_entry = True

        prev_idx = self._prev_idx if self._prev_idx is not None else 0
        try:
            prev_stocks_vec = self.env.stocks.copy()
        except Exception:
            prev_stocks_vec = None

        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        curr_idx = getattr(self.env, "day", prev_idx)
        self._prev_idx = curr_idx
        
        # Debug entry attempts
        if attempting_entry:
            info = dict(info or {})
            info["attempting_entry"] = True
            info["action_sent"] = float(action_arr[0] if action_arr.ndim > 0 else action_arr)
            info["stocks_after"] = float(self.env.stocks.sum())

        # Zero out mark-to-market if configured
        if not self.step_mark_to_market:
            # remove base per-step reward; we will add only exit/time penalties/bonuses
            reward = 0.0
            if hasattr(self.env, "gamma_reward"):
                # we cannot easily remove from gamma_reward; but PPO uses returned reward
                pass

        # Optional small time penalty while holding
        if self.time_penalty_per_bar and float(np.maximum(self.env.stocks, 0.0).sum()) > 0.0:
            pen = self.time_penalty_per_bar * getattr(self.env, "reward_scaling", 1.0)
            reward -= pen
            if hasattr(self.env, "gamma_reward"):
                self.env.gamma_reward -= pen

        penalty_cash, penalty_scaled = self._apply_overnight_penalty(prev_idx, curr_idx)
        if penalty_cash > 0.0:
            self.env.amount -= penalty_cash
            self.env.total_asset -= penalty_cash
            reward -= penalty_scaled
            if hasattr(self.env, "gamma_reward"):
                self.env.gamma_reward -= penalty_scaled
            info = dict(info or {})
            info["overnight_penalty"] = penalty_cash
            info["overnight_penalty_usd"] = penalty_cash * self.log_multiplier

        proceeds = self._enforce_position_cap(curr_idx)
        if proceeds > 0.0:
            info = dict(info or {})
            info["position_cap_enforced"] = proceeds

        if prev_stocks_vec is not None:
            self._maybe_set_tp_sl_on_entry(prev_stocks_vec, curr_idx)
            # Detect and log entry (flat -> long)
            prev_expo = float(np.maximum(prev_stocks_vec, 0.0).sum())
            new_expo = float(np.maximum(self.env.stocks, 0.0).sum())
            if prev_expo <= 0.0 and new_expo > 0.0 and self._entry_price is not None:
                info = dict(info or {})
                info["entry"] = True
                info["entry_price"] = float(self._entry_price)
                notional_idx = float(self._entry_price * new_expo)  # index units
                info["entry_notional_usd"] = notional_idx * self.log_multiplier
                info["entry_step"] = int(self.env.day)
                self._ep_entries += 1
            bonus_cash, bonus_scaled = self._apply_entry_bonus(prev_stocks_vec, curr_idx)
            if bonus_cash > 0.0:
                reward += bonus_scaled
                if hasattr(self.env, "gamma_reward"):
                    self.env.gamma_reward += bonus_scaled
                info = dict(info or {})
                info["entry_bonus"] = bonus_cash
                info["entry_bonus_usd"] = bonus_cash * self.log_multiplier

        self._maybe_trail_stop(curr_idx)
        exited, extra_scaled, exit_type, realized = self._maybe_exit_tp_sl(curr_idx)
        if exited:
            reward += extra_scaled
            if hasattr(self.env, "gamma_reward"):
                self.env.gamma_reward += extra_scaled
            info = dict(info or {})
            info["tp_sl_exit"] = exit_type
            info["trade_realized_pnl"] = realized
            info["trade_realized_pnl_usd"] = realized * self.log_multiplier

        if terminated or truncated:
            info = dict(info or {})
            info.update({
                "ep_trades": self._ep_trades,
                "ep_wins": self._ep_wins,
                "ep_losses": self._ep_losses,
                "ep_realized_pnl": self._ep_realized_pnl,
                "ep_realized_pnl_usd": self._ep_realized_pnl * self.log_multiplier,
                "ep_entries": self._ep_entries,
            })

        return obs, reward, terminated, truncated, info


# Backward-compatible alias
OvernightPenaltyWrapper = TradingRulesWrapper


