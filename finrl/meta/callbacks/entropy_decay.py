"""Entropy decay callback for PPO training."""

from stable_baselines3.common.callbacks import BaseCallback


class EntropyDecayCallback(BaseCallback):
    """Exponentially decay entropy coefficient during training (slow start, fast end).
    
    Uses progress^3 curve to maintain high exploration for longer,
    then rapidly converge once good strategies are found.
    This encourages exploration early and exploitation later.
    """
    
    def __init__(
        self,
        ent_start: float,
        ent_end: float,
        decay_steps: int,
        verbose: int = 0
    ):
        """
        Args:
            ent_start: Starting entropy coefficient (e.g., 0.5)
            ent_end: Final entropy coefficient (e.g., 0.01)
            decay_steps: Number of timesteps to decay over
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.ent_start = ent_start
        self.ent_end = ent_end
        self.decay_steps = decay_steps
        
    def _on_step(self) -> bool:
        """Update entropy coefficient with step-based decay (6 steps for gradual transitions)."""
        # Calculate which step we're in (6 total steps)
        progress = self.num_timesteps / self.decay_steps

        if progress >= 1.0:
            step_index = 6
        elif progress >= 5/6:
            step_index = 5
        elif progress >= 4/6:
            step_index = 4
        elif progress >= 3/6:
            step_index = 3
        elif progress >= 2/6:
            step_index = 2
        elif progress >= 1/6:
            step_index = 1
        else:
            step_index = 0

        # Linear interpolation based on step
        decay_amount = (step_index / 6.0)
        new_ent = self.ent_start - (self.ent_start - self.ent_end) * decay_amount

        # Update model's entropy coefficient
        self.model.ent_coef = new_ent

        # Log to TensorBoard
        self.logger.record("train/ent_coef_current", new_ent)

        return True

