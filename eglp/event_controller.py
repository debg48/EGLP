"""Event controllers for EGLP framework.

Controllers decide when to broadcast global event signals that gate
local weight updates. An event allows layers to incorporate error information.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional


class EventController(ABC):
    """Abstract base class for event controllers."""
    
    @abstractmethod
    def should_trigger(self, current_loss: float) -> int:
        """Decide whether to trigger an event.
        
        Args:
            current_loss: Current batch loss value.
            
        Returns:
            1 if event should be triggered, 0 otherwise.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset controller state for new epoch/experiment."""
        pass
    
    @property
    @abstractmethod
    def events_triggered(self) -> int:
        """Total number of events triggered."""
        pass


class ThresholdController(EventController):
    """Triggers events when loss exceeds a threshold based on running average.
    
    Event is triggered when: current_loss > mean_loss * threshold_factor
    
    This implements a simple surprise-based triggering mechanism:
    when the loss is unexpectedly high, it signals that global error
    information should be incorporated into local updates.
    """
    
    def __init__(
        self,
        budget: int,
        threshold_factor: float = 1.5,
        warmup_steps: int = 10,
        history_size: int = 100,
    ):
        """Initialize threshold controller.
        
        Args:
            budget: Maximum number of events allowed (communication budget).
            threshold_factor: Multiplier for mean loss to set threshold.
            warmup_steps: Steps before triggering is enabled.
            history_size: Size of rolling loss history window.
        """
        self.initial_budget = budget
        self.budget = budget
        self.threshold_factor = threshold_factor
        self.warmup_steps = warmup_steps
        self.history_size = history_size
        
        self.loss_history: List[float] = []
        self._events_triggered = 0
        self._step = 0
    
    def should_trigger(self, current_loss: float) -> int:
        """Trigger event if loss exceeds threshold and budget remains."""
        self._step += 1
        
        # Update history
        self.loss_history.append(current_loss)
        if len(self.loss_history) > self.history_size:
            self.loss_history.pop(0)
        
        # Don't trigger during warmup
        if self._step <= self.warmup_steps:
            return 0
        
        # Check budget
        if self.budget <= 0:
            return 0
        
        # Compute threshold
        mean_loss = sum(self.loss_history) / len(self.loss_history)
        threshold = mean_loss * self.threshold_factor
        
        # Trigger if loss exceeds threshold
        if current_loss > threshold:
            self.budget -= 1
            self._events_triggered += 1
            return 1
        
        return 0
    
    def reset(self) -> None:
        """Reset controller for new experiment."""
        self.budget = self.initial_budget
        self.loss_history = []
        self._events_triggered = 0
        self._step = 0
    
    @property
    def events_triggered(self) -> int:
        return self._events_triggered
    
    @property
    def budget_remaining(self) -> int:
        return self.budget


class FixedRateController(EventController):
    """Triggers events at a fixed rate (for ablation studies).
    
    Each step has probability `rate` of triggering an event.
    Useful for comparing adaptive (ThresholdController) vs. fixed-rate events.
    """
    
    def __init__(self, rate: float, budget: Optional[int] = None, seed: int = 42):
        """Initialize fixed rate controller.
        
        Args:
            rate: Probability of event per step, in [0, 1].
            budget: Optional maximum events (None = unlimited).
            seed: Random seed for reproducibility.
        """
        if not 0 <= rate <= 1:
            raise ValueError(f"Rate must be in [0, 1], got {rate}")
        
        self.rate = rate
        self.initial_budget = budget
        self.budget = budget
        self.seed = seed
        self._rng = random.Random(seed)
        self._events_triggered = 0
    
    def should_trigger(self, current_loss: float) -> int:
        """Trigger event based on fixed probability."""
        # Check budget if set
        if self.budget is not None and self.budget <= 0:
            return 0
        
        # Random trigger based on rate
        if self._rng.random() < self.rate:
            if self.budget is not None:
                self.budget -= 1
            self._events_triggered += 1
            return 1
        
        return 0
    
    def reset(self) -> None:
        """Reset controller for new experiment."""
        self.budget = self.initial_budget
        self._rng = random.Random(self.seed)
        self._events_triggered = 0
    
    @property
    def events_triggered(self) -> int:
        return self._events_triggered


class AlwaysOnController(EventController):
    """Always triggers events (for pure local learning baseline)."""
    
    def __init__(self):
        self._events_triggered = 0
    
    def should_trigger(self, current_loss: float) -> int:
        self._events_triggered += 1
        return 1
    
    def reset(self) -> None:
        self._events_triggered = 0
    
    @property
    def events_triggered(self) -> int:
        return self._events_triggered


class NeverController(EventController):
    """Never triggers events (for comparison/debugging)."""
    
    def should_trigger(self, current_loss: float) -> int:
        return 0
    
    def reset(self) -> None:
        pass
    
    @property
    def events_triggered(self) -> int:
        return 0
