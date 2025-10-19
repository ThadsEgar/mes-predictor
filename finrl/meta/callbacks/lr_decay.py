"""Learning rate decay callback for PPO training."""

from stable_baselines3.common.callbacks import BaseCallback


class LearningRateDecayCallback(BaseCallback):
    """Exponentially decay learning rate during training (slow start, fast end).
    
    Uses progress^3 curve to maintain high learning rate for exploration,
    then rapidly reduce for fine-tuning convergence.
    This helps stabilize training as the model converges.
    """
    
    def __init__(
        self,
        lr_start: float,
        lr_end: float,
        decay_steps: int,
        verbose: int = 0
    ):
        """
        Args:
            lr_start: Starting learning rate (e.g., 1e-4)
            lr_end: Final learning rate (e.g., 1e-6)
            decay_steps: Number of timesteps to decay over
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.decay_steps = decay_steps
        
    def _on_step(self) -> bool:
        """Update learning rate with exponential decay (slow start, fast end)."""
        if self.num_timesteps >= self.decay_steps:
            # Decay complete, use end value
            new_lr = self.lr_end
        else:
            # Exponential decay: slow at start, fast at end
            # Using progress^3 for smooth acceleration
            progress = self.num_timesteps / self.decay_steps
            decay_factor = progress ** 3  # Cube makes it stay high longer, then drop fast
            new_lr = self.lr_start - (self.lr_start - self.lr_end) * decay_factor
        
        # Update model's learning rate
        self.model.learning_rate = new_lr
        
        # Log to TensorBoard
        self.logger.record("train/learning_rate_current", new_lr)
        
        return True

