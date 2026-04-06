"""
Session construction from interaction events
"""
import pandas as pd
from typing import List, Dict
from collections import defaultdict
from datetime import timedelta


class SessionBuilder:
    """Build user sessions from interaction events"""
    
    def __init__(self, gap_minutes: int = 30, min_length: int = 2):
        self.gap_minutes = gap_minutes
        self.min_length = min_length
        
    def build_sessions(self, events: List[Dict]) -> Dict[str, List[List[Dict]]]:
        """
        Group events into sessions by user and time gaps
        
        Returns: {user_id: [session1, session2, ...]}
        """
        # Sort events by user and time
        events_df = pd.DataFrame(events)
        events_df = events_df.sort_values(['user_id', 'timestamp'])
        
        sessions_by_user = defaultdict(list)
        
        for user_id, user_events in events_df.groupby('user_id'):
            user_sessions = self._split_into_sessions(user_events)
            
            # Filter short sessions
            user_sessions = [s for s in user_sessions if len(s) >= self.min_length]
            
            if user_sessions:
                sessions_by_user[user_id] = user_sessions
        
        return dict(sessions_by_user)
    
    def _split_into_sessions(self, user_events: pd.DataFrame) -> List[List[Dict]]:
        """
        Split user events into sessions based on time gaps
        """
        sessions = []
        current_session = []
        prev_time = None
        
        for _, event in user_events.iterrows():
            current_time = event['timestamp']
            
            # Check if new session should start
            if prev_time is not None:
                time_gap = (current_time - prev_time).total_seconds() / 60
                
                if time_gap > self.gap_minutes:
                    # Save current session and start new one
                    if current_session:
                        sessions.append(current_session)
                    current_session = []
            
            # Add event to current session
            current_session.append({
                'article_id': event['article_id'],
                'clicked': event['clicked'],
                'timestamp': event['timestamp']
            })
            
            prev_time = current_time
        
        # Add final session
        if current_session:
            sessions.append(current_session)
        
        return sessions
    
    def get_session_stats(self, sessions: Dict[str, List[List[Dict]]]) -> Dict:
        """
        Compute statistics about sessions
        """
        total_sessions = sum(len(user_sessions) for user_sessions in sessions.values())
        total_users = len(sessions)
        
        session_lengths = []
        for user_sessions in sessions.values():
            for session in user_sessions:
                session_lengths.append(len(session))
        
        return {
            'total_users': total_users,
            'total_sessions': total_sessions,
            'avg_sessions_per_user': total_sessions / total_users if total_users > 0 else 0,
            'avg_session_length': sum(session_lengths) / len(session_lengths) if session_lengths else 0,
            'min_session_length': min(session_lengths) if session_lengths else 0,
            'max_session_length': max(session_lengths) if session_lengths else 0
        }
