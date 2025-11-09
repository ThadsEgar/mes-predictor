"""Inactivity penalty callback for RL training."""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class InactivityPenaltyCallback(BaseCallback):
    """Apply inactivity penalty if agent doesn't trade for extended period.

    Monitors trading activity across all environments. If no trades occur
    for a specified number of timesteps (e.g., 2 trading days), activates
    an inactivity penalty to encourage the agent to explore more actively.

    This serves as a fallback to prevent the agent from becoming too passive.
    """

    def __init__(
        self,
        inactivity_threshold: int = 1000,  # ~2 trading days with minute data
        penalty_value: float = 0.0001,
        verbose: int = 0
    ):
        """
        Args:
            inactivity_threshold: Number of timesteps without trades before penalty activates
                                 (e.g., 1000 for ~2 trading days with minute data)
            penalty_value: Penalty to apply when inactive (e.g., 0.0001)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.inactivity_threshold = inactivity_threshold
        self.penalty_value = penalty_value
        self.last_trade_timestep = 0
        self.penalty_active = False

    def _on_step(self) -> bool:
        """Monitor trading activity and activate penalty if needed."""
        # Check if any environment executed a trade in this step
        # We need to access the info dicts from the rollout buffer
        # For vectorized envs, we check each environment's info

        trade_occurred = False

        # Access the most recent info dicts
        # Note: self.locals contains the step info
        if 'infos' in self.locals:
            infos = self.locals['infos']
            for info in infos:
                # Check if this step had a trade (sell action)
                if 'action' in info and 'sell' in info.get('action', ''):
                    trade_occurred = True
                    break

        # Update last trade timestamp
        if trade_occurred:
            self.last_trade_timestep = self.num_timesteps
            # Deactivate penalty if it was active
            if self.penalty_active:
                self.penalty_active = False
                if self.verbose > 0:
                    print(f"Step {self.num_timesteps}: Trade detected, deactivating inactivity penalty")
                self._set_inactivity_penalty(0.0)

        # Check if we should activate the penalty
        timesteps_since_last_trade = self.num_timesteps - self.last_trade_timestep

        if not self.penalty_active and timesteps_since_last_trade > self.inactivity_threshold:
            self.penalty_active = True
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: No trades for {timesteps_since_last_trade} steps, "
                      f"activating inactivity penalty ({self.penalty_value})")
            self._set_inactivity_penalty(self.penalty_value)

        # Log to TensorBoard
        self.logger.record("inactivity/timesteps_since_last_trade", timesteps_since_last_trade)
        self.logger.record("inactivity/penalty_active", int(self.penalty_active))
        self.logger.record("inactivity/penalty_value", self.penalty_value if self.penalty_active else 0.0)

        return True

    def _set_inactivity_penalty(self, penalty: float):
        """Set the inactivity penalty in all environments."""
        try:
            # Access the wrapped VecEnv
            vec_env = self.model.env
            # If wrapped in VecNormalize, unwrap it
            if hasattr(vec_env, 'venv'):
                vec_env = vec_env.venv

            # Set the attribute in all environments using set_attr
            vec_env.set_attr('inactivity_penalty', penalty)

            if self.verbose > 0:
                print(f"Updated inactivity_penalty to {penalty}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not update inactivity penalty: {e}")
