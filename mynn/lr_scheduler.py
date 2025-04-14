from abc import abstractmethod

import numpy as np


class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    """
    MultiStep Learning Rate Scheduler.
    Decays the learning rate by a factor at specified milestones.
    """
    def __init__(self, optimizer, milestones, gamma):
        super().__init__(optimizer)
        self.milestones = sorted(milestones)  # List of step indices where decay occurs
        self.gamma = gamma  # Decay factor
        self.init_lr = optimizer.init_lr  # Initial learning rate

    def step(self):
        """
        Update the learning rate based on the current step and milestones.
        """
        self.step_count += 1
        # Count how many milestones have passed
        num_decays = sum(self.step_count >= m for m in self.milestones)
        # Compute the new learning rate. Update the optimizer's learning rate
        new_lr = self.init_lr * (self.gamma ** num_decays) 
        self.optimizer.update_lr(new_lr)


class ExponentialLR(scheduler):
    pass