"""Transaction cost curriculum learning callback for RL training."""

from stable_baselines3.common.callbacks import BaseCallback


class TransactionCostCurriculumCallback(BaseCallback):
    """Gradually increase transaction costs during training (curriculum learning).

    Starts with low/zero transaction costs to allow the agent to learn profitable
    strategies, then gradually increases to realistic levels. This helps the agent
    first learn what good trades look like before dealing with cost constraints.

    Uses step-based progression (6 steps) for gradual transitions.
    """

    def __init__(
        self,
        cost_start: float,
        cost_end: float,
        curriculum_steps: int,
        verbose: int = 0
    ):
        """
        Args:
            cost_start: Starting transaction cost in bps (e.g., 0.0)
            cost_end: Final transaction cost in bps (e.g., 1.0)
            curriculum_steps: Number of timesteps to increase over
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.cost_start = cost_start
        self.cost_end = cost_end
        self.curriculum_steps = curriculum_steps

    def _on_step(self) -> bool:
        """Update transaction cost with step-based curriculum (6 steps for gradual transitions)."""
        # Calculate which step we're in (6 total steps)
        progress = self.num_timesteps / self.curriculum_steps

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
        progress_amount = (step_index / 6.0)
        new_cost = self.cost_start + (self.cost_end - self.cost_start) * progress_amount

        # Update transaction cost in all environments
        # VecNormalize wraps the actual VecEnv, so we need to go through .venv
        try:
            # Access the wrapped VecEnv (SubprocVecEnv or DummyVecEnv)
            vec_env = self.model.env
            # If wrapped in VecNormalize, unwrap it
            if hasattr(vec_env, 'venv'):
                vec_env = vec_env.venv

            # Set the attribute in all environments using set_attr
            vec_env.set_attr('transaction_cost_bps', new_cost)

            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Updated transaction_cost_bps to {new_cost:.3f}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not update transaction cost: {e}")

        # Log to TensorBoard
        self.logger.record("curriculum/transaction_cost_bps", new_cost)
        self.logger.record("curriculum/cost_progress", progress_amount)

        return True
