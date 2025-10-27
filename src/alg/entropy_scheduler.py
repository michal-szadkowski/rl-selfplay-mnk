from typing import Dict, Any, Optional


class EntropyScheduler:
    """Handles entropy coefficient scheduling during training."""
    
    def __init__(self, initial_coef: float, schedule: Optional[Dict[str, Any]] = None):
        """
        Initialize entropy scheduler.
        
        Args:
            initial_coef: Initial entropy coefficient
            schedule: Schedule configuration dict with 'type' and 'params' keys.
                     Supported types: 'linear', 'exponential'
        """
        self.initial_coef = initial_coef
        self.current_coef = initial_coef
        self.schedule = schedule
        
    def update(self, current_step: int) -> float:
        """
        Update entropy coefficient based on current training step.
        
        Args:
            current_step: Current training step
            
        Returns:
            Updated entropy coefficient
        """
        if self.schedule is None:
            return self.current_coef
        
        schedule_type = self.schedule.get('type', 'constant')
        params = self.schedule.get('params', {})
        
        if schedule_type == 'linear':
            final_coef = params.get('final_coef', 0.0)
            total_steps = params.get('total_steps', 10_000_000)
            
            if current_step >= total_steps:
                self.current_coef = final_coef
            else:
                progress = current_step / total_steps
                self.current_coef = self.initial_coef * (1 - progress) + final_coef * progress
                
        elif schedule_type == 'exponential':
            decay_rate = params.get('decay_rate', 0.99)
            self.current_coef = self.initial_coef * (decay_rate ** (current_step / 1000))
            
        return self.current_coef