"""
Drift Detector: KS-test + CUSUM for data and model distribution monitoring.

Fix 8 (P1): Production drift detection - was only documented, now implemented.
Detects distribution shift in features and predictions to trigger retraining.
"""

import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Kolmogorov-Smirnov test based drift detector.
    
    Compares current feature/prediction distributions against a reference
    distribution. Triggers drift alerts when p-value falls below threshold.
    """
    
    def __init__(self, window_size: int = 1000, threshold: float = 0.05):
        """
        Args:
            window_size: Number of samples in the sliding window
            threshold: KS test p-value threshold for drift detection
        """
        self.window_size = window_size
        self.threshold = threshold
        self.reference: Optional[np.ndarray] = None
        self.current_window: deque = deque(maxlen=window_size)
        self.drift_history: List[Dict] = []
    
    def update_reference(self, distribution: np.ndarray):
        """Set reference distribution (e.g., from training data)"""
        self.reference = np.array(distribution, dtype=np.float64)
        logger.info(f"Reference distribution set: n={len(self.reference)}, "
                     f"mean={self.reference.mean():.4f}, std={self.reference.std():.4f}")
    
    def add_sample(self, value: float):
        """Add a single observation to the current window"""
        self.current_window.append(value)
    
    def add_batch(self, values: np.ndarray):
        """Add a batch of observations to the current window"""
        for v in values:
            self.current_window.append(float(v))
    
    def detect(self, current: Optional[np.ndarray] = None) -> Dict:
        """
        Detect drift between reference and current distributions.
        
        Args:
            current: Current distribution to test. If None, uses sliding window.
            
        Returns:
            Dict with drift_detected, ks_statistic, p_value, and diagnostics
        """
        if self.reference is None:
            return {
                'drift_detected': False,
                'reason': 'no_reference_set'
            }
        
        if current is None:
            if len(self.current_window) < 50:
                return {
                    'drift_detected': False,
                    'reason': f'insufficient_samples ({len(self.current_window)}/50)'
                }
            current = np.array(list(self.current_window))
        
        if len(current) < 10:
            return {
                'drift_detected': False,
                'reason': f'insufficient_samples ({len(current)}/10)'
            }
        
        stat, p_value = ks_2samp(self.reference, current)
        
        drift_detected = p_value < self.threshold
        
        result = {
            'drift_detected': drift_detected,
            'ks_statistic': float(stat),
            'p_value': float(p_value),
            'threshold': self.threshold,
            'reference_mean': float(self.reference.mean()),
            'reference_std': float(self.reference.std()),
            'current_mean': float(current.mean()),
            'current_std': float(current.std()),
            'current_n': len(current),
            'mean_shift': float(abs(current.mean() - self.reference.mean())),
        }
        
        if drift_detected:
            logger.warning(
                f"DRIFT DETECTED: KS={stat:.4f}, p={p_value:.6f}, "
                f"mean_shift={result['mean_shift']:.4f}"
            )
            self.drift_history.append(result)
        
        return result
    
    def get_drift_summary(self) -> Dict:
        """Get summary of all detected drift events"""
        return {
            'total_drift_events': len(self.drift_history),
            'recent_drifts': self.drift_history[-5:] if self.drift_history else [],
            'window_size': self.window_size,
            'threshold': self.threshold,
            'current_window_size': len(self.current_window),
        }


class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) change-point detector.
    
    Detects sustained shifts in the mean of a time series.
    More sensitive to gradual drift than KS-test.
    """
    
    def __init__(self, target_mean: float = 0.0, threshold: float = 5.0,
                 drift_rate: float = 0.5):
        """
        Args:
            target_mean: Expected mean (from training data)
            threshold: Decision threshold for detecting change
            drift_rate: Minimum detectable drift magnitude
        """
        self.target_mean = target_mean
        self.threshold = threshold
        self.drift_rate = drift_rate
        self.s_pos = 0.0  # Upper CUSUM
        self.s_neg = 0.0  # Lower CUSUM
        self.observations = 0
    
    def update(self, value: float) -> Dict:
        """
        Process one observation and check for drift.
        
        Returns:
            Dict with drift status and CUSUM values
        """
        self.observations += 1
        deviation = value - self.target_mean
        
        self.s_pos = max(0, self.s_pos + deviation - self.drift_rate)
        self.s_neg = max(0, self.s_neg - deviation - self.drift_rate)
        
        drift_up = self.s_pos > self.threshold
        drift_down = self.s_neg > self.threshold
        
        result = {
            'drift_detected': drift_up or drift_down,
            'direction': 'up' if drift_up else ('down' if drift_down else 'none'),
            's_pos': float(self.s_pos),
            's_neg': float(self.s_neg),
            'threshold': self.threshold,
            'observations': self.observations,
        }
        
        # Reset after detection
        if drift_up:
            self.s_pos = 0.0
        if drift_down:
            self.s_neg = 0.0
        
        return result
    
    def reset(self):
        """Reset CUSUM accumulators"""
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.observations = 0


class MultiFeatureDriftMonitor:
    """
    Monitor drift across multiple features simultaneously.
    Production usage: attach to serving pipeline.
    """
    
    def __init__(self, feature_names: List[str],
                 window_size: int = 1000, threshold: float = 0.05):
        self.detectors = {
            name: DriftDetector(window_size=window_size, threshold=threshold)
            for name in feature_names
        }
    
    def set_reference(self, feature_name: str, reference: np.ndarray):
        """Set reference distribution for a feature"""
        if feature_name in self.detectors:
            self.detectors[feature_name].update_reference(reference)
    
    def add_observation(self, features: Dict[str, float]):
        """Add one observation (all features)"""
        for name, value in features.items():
            if name in self.detectors:
                self.detectors[name].add_sample(value)
    
    def check_all(self) -> Dict[str, Dict]:
        """Run drift detection on all features"""
        results = {}
        any_drift = False
        
        for name, detector in self.detectors.items():
            result = detector.detect()
            results[name] = result
            if result.get('drift_detected'):
                any_drift = True
        
        return {
            'any_drift_detected': any_drift,
            'features': results
        }
