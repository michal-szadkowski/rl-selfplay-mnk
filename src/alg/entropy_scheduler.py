from typing import Dict, Any, Optional


class EntropyScheduler:
    """Handles entropy coefficient scheduling during training."""
    
    def __init__(self, initial_coef: float, schedule: Optional[Dict[str, Any]] = None,
                 num_envs: int = None, n_steps: int = None):
        """
        Initialize entropy scheduler.
        
        Args:
            initial_coef: Initial entropy coefficient
            schedule: Schedule configuration dict with 'type' and 'params' keys.
                     Supported types: 'linear', 'exponential'
            num_envs: Number of parallel environments (for step conversion)
            n_steps: Number of steps per rollout (for step conversion)
        """
        self.initial_coef = initial_coef
        self.current_coef = initial_coef
        self.schedule = schedule
        self.current_step = 0
        self.steps_per_iteration = num_envs * n_steps if num_envs and n_steps else None
        
    def step(self) -> None:
        """
        Update entropy coefficient based on current training step.
        """
        if self.schedule is None:
            return
        
        self.current_step += 1  # increment iteration count
        
        # Convert iterations to environment steps if conversion info available
        if self.steps_per_iteration:
            current_env_steps = self.current_step * self.steps_per_iteration
        else:
            current_env_steps = self.current_step  # fallback to iterations
        
        schedule_type = self.schedule.get('type', 'constant')
        params = self.schedule.get('params', {})
        
        if schedule_type == 'linear':
            final_coef = params.get('final_coef', 0.0)
            total_steps = params.get('total_steps', 10_000_000)
            
            if current_env_steps >= total_steps:
                self.current_coef = final_coef
            else:
                progress = current_env_steps / total_steps
                self.current_coef = self.initial_coef * (1 - progress) + final_coef * progress
                
        elif schedule_type == 'exponential':
            decay_rate = params.get('decay_rate', 0.99)
            self.current_coef = self.initial_coef * (decay_rate ** (current_env_steps / 1000))
    
    def get_last_coef(self) -> float:
        """Get the current entropy coefficient."""
        return self.current_coef