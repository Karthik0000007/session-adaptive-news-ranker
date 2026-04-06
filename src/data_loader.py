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
