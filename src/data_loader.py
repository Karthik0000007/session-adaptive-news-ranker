"""
Data loading utilities for MIND dataset
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class MINDDataLoader:
    """Load and parse MIND dataset files"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_behaviors(self, filename: str = "behaviors.tsv") -> pd.DataFrame:
        """
        Load behaviors.tsv
        
        Columns: impression_id, user_id, time, history, impressions
        """
        filepath = self.data_dir / filename
        
        df = pd.read_csv(
            filepath,
            sep='\t',
            header=None,
            names=['impression_id', 'user_id', 'time', 'history', 'impressions']
        )
        
        # Parse timestamp
        df['time'] = pd.to_datetime(df['time'])
        
        return df
    
    def load_news(self, filename: str = "news.tsv") -> pd.DataFrame:
        """
        Load news.tsv
        
        Columns: news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
        """
        filepath = self.data_dir / filename
        
        df = pd.read_csv(
            filepath,
            sep='\t',
            header=None,
            names=['news_id', 'category', 'subcategory', 'title', 'abstract', 
                   'url', 'title_entities', 'abstract_entities']
        )
        
        return df
    
    def parse_impressions(self, behaviors_df: pd.DataFrame) -> List[Dict]:
        """
        Convert impression-based format to event-based format
        
        Input: behaviors.tsv rows
        Output: List of interaction events
        """
        events = []
        
        for _, row in behaviors_df.iterrows():
            user_id = row['user_id']
            timestamp = row['time']
            impressions = row['impressions']
            
            if pd.isna(impressions):
                continue
            
            # Parse impressions: "N1-1 N2-0 N3-1"
            for impression in impressions.split():
                parts = impression.split('-')
                if len(parts) != 2:
                    continue
                    
                article_id, clicked = parts
                
                events.append({
                    'user_id': user_id,
                    'timestamp': timestamp,
                    'article_id': article_id,
                    'clicked': int(clicked)
                })
        
        return events
    
    def get_article_metadata(self, news_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Create article_id -> metadata mapping
        """
        metadata = {}
        
        for _, row in news_df.iterrows():
            metadata[row['news_id']] = {
                'category': row['category'],
                'subcategory': row['subcategory'],
                'title': row['title']
            }
        
        return metadata

    @staticmethod
    def build_retention_labels(events: List[Dict],
                                session_gap_minutes: float = 5.0) -> List[Dict]:
        """
        Compute session-continuation labels from interaction events.
        
        Label = 1 if user has another interaction within session_gap_minutes
        Label = 0 if session ends after this interaction
        
        This produces labels that are genuinely different from click labels,
        fixing the broken 4-objective system.
        
        Args:
            events: List of interaction dicts with 'user_id', 'timestamp' keys
            session_gap_minutes: Max gap (minutes) for session continuation
            
        Returns:
            Same events list with 'label_retention' field added
        """
        import numpy as np
        
        # Sort events by user and time
        events_df = pd.DataFrame(events)
        if 'timestamp' not in events_df.columns or len(events_df) == 0:
            for e in events:
                e['label_retention'] = 0
            return events
        
        events_df = events_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        events_df['label_retention'] = 0
        
        for user_id, group in events_df.groupby('user_id'):
            indices = group.index.tolist()
            timestamps = group['timestamp'].values
            
            for i in range(len(indices) - 1):
                current_ts = pd.Timestamp(timestamps[i])
                next_ts = pd.Timestamp(timestamps[i + 1])
                gap_minutes = (next_ts - current_ts).total_seconds() / 60.0
                
                if gap_minutes <= session_gap_minutes:
                    events_df.loc[indices[i], 'label_retention'] = 1
            # Last event in user sequence: label = 0 (session ends)
        
        # Map back to events list
        for idx, event in enumerate(events):
            matching = events_df[
                (events_df['user_id'] == event['user_id']) &
                (events_df['article_id'] == event['article_id'])
            ]
            if len(matching) > 0:
                event['label_retention'] = int(matching.iloc[0]['label_retention'])
            else:
                event['label_retention'] = 0
        
        return events
