"""
Rule-Based Weight Adaptation: Session-aware weight adjustment
"""
import numpy as np
from typing import Dict, List, Tuple


class WeightAdapter:
    """Adaptive weight selection based on session state"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Thresholds
        self.early_session_threshold = config.get('early_session_threshold', 3)
        self.high_engagement_threshold = config.get('high_engagement_threshold', 30.0)
        self.fatigue_threshold = config.get('fatigue_threshold', 0.6)
        
        # Smoothing factor
        self.alpha = config.get('smoothing_alpha', 0.7)
        
        # Previous weights (for smoothing)
        self.previous_weights = None
        
        # Default weights
        self.default_weights = [0.4, 0.3, 0.2, 0.1]  # [E, R, D, N]
    
    def get_weights(self, session_state: Dict, smooth: bool = True) -> List[float]:
        """
        Get adaptive weights based on session state
        
        Args:
            session_state: Session features
            smooth: Apply smoothing with previous weights
        
        Returns: [w_engagement, w_retention, w_diversity, w_novelty]
        """
        # Apply rules
        weights = self._apply_rules(session_state)
        
        # Smooth transitions
        if smooth and self.previous_weights is not None:
            weights = self._smooth_weights(weights, self.previous_weights)
        
        # Ensure constraints
        weights = self._enforce_constraints(weights)
        
        # Update previous
        self.previous_weights = weights
        
        return weights
    
    def _apply_rules(self, session_state: Dict) -> List[float]:
        """
        Apply rule-based logic to determine weights
        
        Rules (in priority order):
        1. Early session: Boost diversity + novelty (prevent bounce)
        2. High engagement: Boost engagement (maximize time)
        3. Fatigue detected: Boost novelty + retention (re-engage)
        4. Default: Balanced weights
        """
        session_length = session_state.get('session_length', 0)
        avg_dwell = session_state.get('avg_dwell_time', 0)
        fatigue = session_state.get('fatigue_score', 0)
        
        # Rule 1: Early session (cold start)
        if session_length < self.early_session_threshold:
            return [0.25, 0.20, 0.35, 0.20]  # Boost diversity + novelty
        
        # Rule 2: High engagement detected
        if avg_dwell > self.high_engagement_threshold:
            return [0.50, 0.25, 0.15, 0.10]  # Maximize engagement
        
        # Rule 3: Fatigue / boredom
        if fatigue > self.fatigue_threshold:
            return [0.20, 0.30, 0.25, 0.25]  # Boost novelty + retention
        
        # Rule 4: Default
        return self.default_weights

    def _smooth_weights(self,
                        new_weights: List[float],
                        prev_weights: List[float]) -> List[float]:
        """
        Smooth weight transitions to avoid abrupt changes

        w_final = alpha * w_new + (1 - alpha) * w_prev
        """
        new_array = np.array(new_weights)
        prev_array = np.array(prev_weights)

        smoothed = self.alpha * new_array + (1 - self.alpha) * prev_array

        return smoothed.tolist()

    def _enforce_constraints(self, weights: List[float]) -> List[float]:
        """
        Enforce weight constraints:
        - All weights >= 0
        - Sum of weights = 1
        """
        weights_array = np.array(weights)

        # Ensure non-negative
        weights_array = np.maximum(weights_array, 0)

        # Normalize to sum to 1
        total = weights_array.sum()
        if total > 0:
            weights_array = weights_array / total
        else:
            weights_array = np.array(self.default_weights)

        return weights_array.tolist()

    def get_rule_explanation(self, session_state: Dict) -> str:
        """
        Get human-readable explanation of which rule was applied

        Useful for debugging and interpretability
        """
        session_length = session_state.get('session_length', 0)
        avg_dwell = session_state.get('avg_dwell_time', 0)
        fatigue = session_state.get('fatigue_score', 0)

        if session_length < self.early_session_threshold:
            return f"Early session (length={session_length}): Boosting diversity + novelty"

        if avg_dwell > self.high_engagement_threshold:
            return f"High engagement (dwell={avg_dwell:.1f}s): Maximizing engagement"

        if fatigue > self.fatigue_threshold:
            return f"Fatigue detected (score={fatigue:.2f}): Boosting novelty + retention"

        return "Default: Balanced weights"

    def reset(self):
        """Reset previous weights (e.g., for new session)"""
        self.previous_weights = None

    def get_baseline_weights(self) -> List[float]:
        """Get fixed baseline weights for comparison"""
        return self.default_weights.copy()


class WeightStrategy:
    """Enum-like class for weight strategies"""
    FIXED = "fixed"
    RULE_BASED = "rule_based"
    BANDIT = "bandit"  # For Phase 6
