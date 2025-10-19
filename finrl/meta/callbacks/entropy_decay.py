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
        """Update entropy coefficient with exponential decay (slow start, fast end)."""
        if self.num_timesteps >= self.decay_steps:
            # Decay complete, use end value
            new_ent = self.ent_end
        else:
            # Exponential decay: slow at start, fast at end
            # Using progress^3 for smooth acceleration
            progress = self.num_timesteps / self.decay_steps
            decay_factor = progress ** 3  # Cube makes it stay high longer, then drop fast
            new_ent = self.ent_start - (self.ent_start - self.ent_end) * decay_factor
        
        # Update model's entropy coefficient
        self.model.ent_coef = new_ent
        
        # Log to TensorBoard
        self.logger.record("train/ent_coef_current", new_ent)
        
        return True

