"""
Extract session state features for training
"""
import numpy as np
from typing import List, Dict
from collections import Counter


class FeatureExtractor:
    """Extract session-aware features from interaction sequences"""
    
    def __init__(self, config: Dict):
        self.recent_k = config['recent_categories_k']
        self.epsilon = config['entropy_smoothing']
    
    def extract_session_state(self, 
                             session_events: List[Dict], 
                             current_idx: int,
                             article_metadata: Dict) -> Dict:
        """
        Extract session state features at a given point in time
        
        This represents what the system "knows" at step t
        """
        if current_idx == 0:
            return self._get_initial_state(session_events[0])
        
        # Look at events up to current point
        past_events = session_events[:current_idx + 1]
        
        # Basic interaction metrics
        session_length = len(past_events)
        avg_dwell = np.mean([e['dwell_time'] for e in past_events])
        
        # Click behavior
        clicks = [e for e in past_events if e['clicked'] == 1]
        click_rate = len(clicks) / session_length
        skip_rate = 1 - click_rate
        
        # Category diversity
        clicked_categories = [
            article_metadata.get(e['article_id'], {}).get('category', 'unknown')
            for e in clicks
        ]
        click_entropy = self._compute_entropy(clicked_categories)
        
        # Recent categories (last K)
        recent_categories = clicked_categories[-self.recent_k:]
        
        # Temporal features
        timestamp = past_events[-1]['timestamp']
        time_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Total session time
        session_duration = (past_events[-1]['timestamp'] - past_events[0]['timestamp']).total_seconds() / 60
        
        return {
            'session_length': session_length,
            'avg_dwell_time': round(avg_dwell, 2),
            'click_rate': round(click_rate, 3),
            'skip_rate': round(skip_rate, 3),
            'click_entropy': round(click_entropy, 3),
            'recent_categories': recent_categories,
            'time_of_day': time_of_day,
            'day_of_week': day_of_week,
            'session_duration_min': round(session_duration, 2)
        }
    
    def _get_initial_state(self, first_event: Dict) -> Dict:
        """Initial state at session start"""
        return {
            'session_length': 1,
            'avg_dwell_time': first_event.get('dwell_time', 0),
            'click_rate': float(first_event['clicked']),
            'skip_rate': 1 - float(first_event['clicked']),
            'click_entropy': 0.0,
            'recent_categories': [],
            'time_of_day': first_event['timestamp'].hour,
            'day_of_week': first_event['timestamp'].weekday(),
            'session_duration_min': 0.0
        }
    
    def _compute_entropy(self, items: List[str]) -> float:
        """
        Compute Shannon entropy of item distribution
        
        High entropy = diverse exploration
        Low entropy = repetitive behavior
        """
        if not items:
            return 0.0
        
        counts = Counter(items)
        total = len(items)
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * np.log(p + self.epsilon)
        
        return entropy
    
    def create_training_sample(self,
                              user_id: str,
                              article_id: str,
                              session_state: Dict,
                              article_metadata: Dict,
                              label_click: int,
                              label_dwell: float) -> Dict:
        """
        Create a single training sample
        
        Format ready for model training
        """
        article_meta = article_metadata.get(article_id, {})
        
        return {
            # Identifiers
            'user_id': user_id,
            'article_id': article_id,
            
            # Session features
            'session_length': session_state['session_length'],
            'avg_dwell_time': session_state['avg_dwell_time'],
            'click_rate': session_state['click_rate'],
            'skip_rate': session_state['skip_rate'],
            'click_entropy': session_state['click_entropy'],
            'time_of_day': session_state['time_of_day'],
            'day_of_week': session_state['day_of_week'],
            'session_duration_min': session_state['session_duration_min'],
            
            # Article features
            'article_category': article_meta.get('category', 'unknown'),
            'article_subcategory': article_meta.get('subcategory', 'unknown'),
            
            # Labels
            'label_click': label_click,
            'label_dwell': label_dwell
        }
