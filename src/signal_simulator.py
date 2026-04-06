"""
Simulate missing behavioral signals (dwell time, fatigue, etc.)
"""
import numpy as np
from typing import List, Dict


class SignalSimulator:
    """Simulate realistic behavioral signals not present in MIND"""
    
    def __init__(self, config: Dict):
        self.clicked_mean = config['clicked_mean']
        self.clicked_sigma = config['clicked_sigma']
        self.not_clicked_min = config['not_clicked_min']
        self.not_clicked_max = config['not_clicked_max']
        
        np.random.seed(42)  # Reproducibility
    
    def simulate_dwell_time(self, clicked: int) -> float:
        """
        Simulate dwell time based on click behavior
        
        Clicked items: log-normal distribution (longer engagement)
        Not clicked: uniform distribution (brief scan)
        """
        if clicked == 1:
            # Log-normal: realistic for reading time
            dwell = np.random.lognormal(self.clicked_mean, self.clicked_sigma)
            # Cap at reasonable maximum (10 minutes)
            dwell = min(dwell * 60, 600)  # Convert to seconds
        else:
            # Uniform: quick scan
            dwell = np.random.uniform(self.not_clicked_min, self.not_clicked_max)
        
        return round(dwell, 2)
    
    def add_signals_to_session(self, session: List[Dict]) -> List[Dict]:
        """
        Add simulated signals to all events in a session
        """
        enriched_session = []
        
        for event in session:
            enriched_event = event.copy()
            enriched_event['dwell_time'] = self.simulate_dwell_time(event['clicked'])
            enriched_session.append(enriched_event)
        
        return enriched_session
    
    def compute_fatigue_score(self, session_events: List[Dict], current_idx: int) -> float:
        """
        Compute fatigue score at a given point in session
        
        fatigue = 0.5 * (1 - normalized_dwell) + 0.5 * skip_rate
        """
        if current_idx == 0:
            return 0.0
        
        # Look at events up to current point
        past_events = session_events[:current_idx + 1]
        
        # Average dwell time (normalized by 60 seconds)
        avg_dwell = np.mean([e['dwell_time'] for e in past_events])
        normalized_dwell = min(avg_dwell / 60.0, 1.0)
        
        # Skip rate
        skips = sum(1 for e in past_events if e['clicked'] == 0)
        skip_rate = skips / len(past_events)
        
        fatigue = 0.5 * (1 - normalized_dwell) + 0.5 * skip_rate
        
        return round(fatigue, 3)
