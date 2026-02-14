"""Event controllers for EGLP framework.

Controllers decide when to broadcast global event signals that gate
local weight updates. Instead of binary gating, controllers return a
continuous error signal that modulates the strength of local updates.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import List, Optional


# Maximum signal magnitude to prevent Hebbian updates from destabilizing features.
# With local_lr=0.01, a signal of 0.1 gives effective update = 0.001, which is
# small enough to let the output layer converge while still refining features.
DEFAULT_MAX_SIGNAL = 0.1


class EventController(ABC):
    """Abstract base class for event controllers.
    
    Controllers return a continuous float signal:
    - 0.0 = no update
    - positive float = strength of error modulation (capped to max_signal)
    """
    
    @abstractmethod
    def should_trigger(self, current_loss: float) -> float:
        """Decide whether to trigger an event and with what strength.
        
        Args:
            current_loss: Current batch loss value.
            
        Returns:
            Continuous error signal. 0.0 = no event, positive = modulation strength.
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
    
    Returns a continuous normalized surprise signal:
        signal = (current_loss - mean_loss) / (std_loss + ε)
    when current_loss > mean_loss * threshold_factor.
    
    This provides both:
    - WHEN to update (surprise-based gating)
    - HOW MUCH to update (magnitude of surprise)
    """
    
    def __init__(
        self,
        budget: int,
        threshold_factor: float = 1.1,
        warmup_steps: int = 10,
        history_size: int = 100,
        max_signal: float = DEFAULT_MAX_SIGNAL,
    ):
        """Initialize threshold controller.
        
        Args:
            budget: Maximum number of events allowed (communication budget).
            threshold_factor: Multiplier for mean loss to set threshold.
                             Default lowered from 1.5 to 1.1 so events
                             actually fire in local-learning settings where
                             loss variance is lower than in backprop.
            warmup_steps: Steps before triggering is enabled.
            history_size: Size of rolling loss history window.
            max_signal: Maximum magnitude of the error signal.
        """
        self.initial_budget = budget
        self.budget = budget
        self.threshold_factor = threshold_factor
        self.warmup_steps = warmup_steps
        self.history_size = history_size
        self.max_signal = max_signal
        
        self.loss_history: List[float] = []
        self._events_triggered = 0
        self._step = 0
    
    def should_trigger(self, current_loss: float) -> float:
        """Trigger event if loss exceeds threshold, returning normalized surprise."""
        self._step += 1
        
        # Handle NaN/Inf loss
        if math.isnan(current_loss) or math.isinf(current_loss):
            return 0.0
        
        # Update history
        self.loss_history.append(current_loss)
        if len(self.loss_history) > self.history_size:
            self.loss_history.pop(0)
        
        # Don't trigger during warmup
        if self._step <= self.warmup_steps:
            return 0.0
        
        # Check budget
        if self.budget <= 0:
            return 0.0
        
        # Compute threshold
        mean_loss = sum(self.loss_history) / len(self.loss_history)
        threshold = mean_loss * self.threshold_factor
        
        # Compute standard deviation for normalization
        if len(self.loss_history) > 1:
            variance = sum((l - mean_loss) ** 2 for l in self.loss_history) / len(self.loss_history)
            std_loss = math.sqrt(variance)
        else:
            std_loss = 1.0
        
        # Trigger if loss exceeds threshold
        if current_loss > threshold:
            self.budget -= 1
            self._events_triggered += 1
            # Return normalized surprise signal, capped to max_signal
            signal = (current_loss - mean_loss) / (std_loss + 1e-8)
            return min(max(signal, 0.01), self.max_signal)
        
        return 0.0
    
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
    """Triggers events at a fixed rate with continuous error signal.
    
    Each step has probability `rate` of triggering an event.
    When triggered, returns a continuous error signal proportional
    to the loss relative to the running mean.
    """
    
    def __init__(self, rate: float, budget: Optional[int] = None, seed: int = 42, max_signal: float = DEFAULT_MAX_SIGNAL):
        """Initialize fixed rate controller.
        
        Args:
            rate: Probability of event per step, in [0, 1].
            budget: Optional maximum events (None = unlimited).
            seed: Random seed for reproducibility.
            max_signal: Maximum magnitude of the error signal.
        """
        if not 0 <= rate <= 1:
            raise ValueError(f"Rate must be in [0, 1], got {rate}")
        
        self.rate = rate
        self.initial_budget = budget
        self.budget = budget
        self.seed = seed
        self.max_signal = max_signal
        self._rng = random.Random(seed)
        self._events_triggered = 0
        self._loss_sum = 0.0
        self._loss_count = 0
    
    def should_trigger(self, current_loss: float) -> float:
        """Trigger event based on fixed probability, returning error signal."""
        # Track running mean loss
        if not (math.isnan(current_loss) or math.isinf(current_loss)):
            self._loss_sum += current_loss
            self._loss_count += 1
        
        # Check budget if set
        if self.budget is not None and self.budget <= 0:
            return 0.0
        
        # Random trigger based on rate
        if self._rng.random() < self.rate:
            if self.budget is not None:
                self.budget -= 1
            self._events_triggered += 1
            
            # Return error signal proportional to loss relative to mean, capped
            if self._loss_count > 0:
                mean_loss = self._loss_sum / self._loss_count
                signal = current_loss / (mean_loss + 1e-8)
                return min(max(signal, 0.01), self.max_signal)
            return self.max_signal
        
        return 0.0
    
    def reset(self) -> None:
        """Reset controller for new experiment."""
        self.budget = self.initial_budget
        self._rng = random.Random(self.seed)
        self._events_triggered = 0
        self._loss_sum = 0.0
        self._loss_count = 0
    
    @property
    def events_triggered(self) -> int:
        return self._events_triggered


class AlwaysOnController(EventController):
    """Always triggers events with continuous error signal.
    
    For pure local learning baseline. Returns error signal based
    on loss relative to running mean.
    """
    
    def __init__(self, max_signal: float = DEFAULT_MAX_SIGNAL):
        self._events_triggered = 0
        self.max_signal = max_signal
        self._loss_sum = 0.0
        self._loss_count = 0
    
    def should_trigger(self, current_loss: float) -> float:
        self._events_triggered += 1
        
        # Track running mean
        if not (math.isnan(current_loss) or math.isinf(current_loss)):
            self._loss_sum += current_loss
            self._loss_count += 1
        
        # Return error signal, capped to max_signal
        if self._loss_count > 1:
            mean_loss = self._loss_sum / self._loss_count
            signal = current_loss / (mean_loss + 1e-8)
            return min(max(signal, 0.01), self.max_signal)
        
        # During early steps, return max_signal as baseline
        return self.max_signal
    
    def reset(self) -> None:
        self._events_triggered = 0
        self._loss_sum = 0.0
        self._loss_count = 0
    
    @property
    def events_triggered(self) -> int:
        return self._events_triggered


class NeverController(EventController):
    """Never triggers events (for comparison/debugging)."""
    
    def should_trigger(self, current_loss: float) -> float:
        return 0.0
    
    def reset(self) -> None:
        pass
    
    @property
    def events_triggered(self) -> int:
        return 0
