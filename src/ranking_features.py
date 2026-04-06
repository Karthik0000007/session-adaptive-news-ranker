"""
Feature Engineering for Ranking Models
Transforms raw data into model-ready features
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Set
from collections import Counter, defaultdict
from datetime import datetime


class RankingFeatureBuilder:
    """Build comprehensive features for ranking models"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # User profile cache
        self.user_profiles = {}
        self.item_stats = {}
        
    def build_user_profiles(self, sessions: Dict[str, List[List[Dict]]]):
        """
        Build user profiles from historical sessions
        
        Computes: total clicks, category preferences, avg dwell, etc.
        """
        print("Building user profiles...")
        
        for user_id, user_sessions in sessions.items():
            clicks = []
            dwells = []
            categories = []
            
            for session in user_sessions:
                for event in session:
                    if event['clicked'] == 1:
                        clicks.append(event['article_id'])
                        dwells.append(event.get('dwell_time', 0))
            
            self.user_profiles[user_id] = {
                'total_clicks': len(clicks),
                'avg_dwell': np.mean(dwells) if dwells else 0,
                'num_sessions': len(user_sessions),
                'clicked_articles': set(clicks)
            }
        
        print(f"  - Built profiles for {len(self.user_profiles)} users")
    
    def build_item_stats(self, article_metadata: Dict[str, Dict], 
                        sessions: Dict[str, List[List[Dict]]]):
        """
        Build item statistics from corpus
        
        Computes: popularity, category distribution, recency
        """
        print("Building item statistics...")
        
        click_counts = Counter()
        impression_counts = Counter()
        
        for user_sessions in sessions.values():
            for session in user_sessions:
                for event in session:
                    article_id = event['article_id']
                    impression_counts[article_id] += 1
                    if event['clicked'] == 1:
                        click_counts[article_id] += 1
        
        for article_id in article_metadata.keys():
            clicks = click_counts.get(article_id, 0)
            impressions = impression_counts.get(article_id, 0)
            
            self.item_stats[article_id] = {
                'popularity': clicks,
                'ctr': clicks / impressions if impressions > 0 else 0,
                'impressions': impressions
            }
        
        print(f"  - Built stats for {len(self.item_stats)} items")
    
    def extract_user_features(self, user_id: str) -> Dict:
        """Extract user-level features"""
        profile = self.user_profiles.get(user_id, {})
        
        return {
            'user_total_clicks': profile.get('total_clicks', 0),
            'user_avg_dwell': profile.get('avg_dwell', 0),
            'user_num_sessions': profile.get('num_sessions', 0)
        }
    
    def extract_item_features(self, article_id: str, 
                             article_metadata: Dict[str, Dict]) -> Dict:
        """Extract item-level features"""
        metadata = article_metadata.get(article_id, {})
        stats = self.item_stats.get(article_id, {})
        
        return {
            'item_category': metadata.get('category', 'unknown'),
            'item_subcategory': metadata.get('subcategory', 'unknown'),
            'item_popularity': stats.get('popularity', 0),
            'item_ctr': stats.get('ctr', 0),
            'item_impressions': stats.get('impressions', 0)
        }
    
    def extract_session_features(self, session_state: Dict) -> Dict:
        """Extract session-level features"""
        return {
            'session_length': session_state.get('session_length', 0),
            'session_avg_dwell': session_state.get('avg_dwell_time', 0),
            'session_click_rate': session_state.get('click_rate', 0),
            'session_skip_rate': session_state.get('skip_rate', 0),
            'session_click_entropy': session_state.get('click_entropy', 0),
            'session_fatigue': session_state.get('fatigue_score', 0),
            'session_time_of_day': session_state.get('time_of_day', 0),
            'session_day_of_week': session_state.get('day_of_week', 0)
        }
    
    def extract_interaction_features(self, 
                                    user_id: str,
                                    article_id: str,
                                    article_metadata: Dict[str, Dict],
                                    clicked_history: List[str]) -> Dict:
        """
        Extract user-item interaction features
        
        These are often the highest signal features
        """
        profile = self.user_profiles.get(user_id, {})
        metadata = article_metadata.get(article_id, {})
        
        # Category affinity
        user_clicked = profile.get('clicked_articles', set())
        category = metadata.get('category', 'unknown')
        
        # Count how many times user clicked this category
        category_clicks = 0
        for clicked_id in user_clicked:
            if article_metadata.get(clicked_id, {}).get('category') == category:
                category_clicks += 1
        
        # Recency features
        has_clicked_before = article_id in user_clicked
        
        # Similarity to recent clicks
        similarity_to_recent = 0.0
        if clicked_history:
            last_clicked_id = clicked_history[-1]
            last_category = article_metadata.get(last_clicked_id, {}).get('category')
            similarity_to_recent = 1.0 if category == last_category else 0.0
        
        return {
            'user_category_affinity': category_clicks,
            'has_clicked_before': int(has_clicked_before),
            'similarity_to_recent': similarity_to_recent,
            'num_recent_clicks': len(clicked_history)
        }
    
    def build_features(self,
                      user_id: str,
                      article_id: str,
                      session_state: Dict,
                      article_metadata: Dict[str, Dict],
                      clicked_history: List[str]) -> Dict:
        """
        Build complete feature vector for ranking
        
        Combines all feature types
        """
        features = {}
        
        # User features
        features.update(self.extract_user_features(user_id))
        
        # Item features
        features.update(self.extract_item_features(article_id, article_metadata))
        
        # Session features
        features.update(self.extract_session_features(session_state))
        
        # Interaction features
        features.update(self.extract_interaction_features(
            user_id, article_id, article_metadata, clicked_history
        ))
        
        return features
    
    def prepare_training_data(self,
                             training_samples: List[Dict],
                             article_metadata: Dict[str, Dict]) -> pd.DataFrame:
        """
        Convert training samples to feature DataFrame
        
        Args:
            training_samples: From Phase 1
            article_metadata: Article metadata
        
        Returns: DataFrame with features and labels
        """
        print("Preparing training data...")
        
        rows = []
        
        for sample in training_samples:
            # Build session state from sample
            session_state = {
                'session_length': sample.get('session_length', 0),
                'avg_dwell_time': sample.get('avg_dwell_time', 0),
                'click_rate': sample.get('click_rate', 0),
                'skip_rate': sample.get('skip_rate', 0),
                'click_entropy': sample.get('click_entropy', 0),
                'fatigue_score': 0.0,  # Not in training samples
                'time_of_day': sample.get('time_of_day', 0),
                'day_of_week': sample.get('day_of_week', 0)
            }
            
            # Build features
            features = self.build_features(
                user_id=sample['user_id'],
                article_id=sample['article_id'],
                session_state=session_state,
                article_metadata=article_metadata,
                clicked_history=[]  # Simplified for now
            )
            
            # Add labels
            features['label_click'] = sample['label_click']
            features['label_dwell'] = sample['label_dwell']
            
            # Add metadata for splitting
            features['user_id'] = sample['user_id']
            features['article_id'] = sample['article_id']
            
            rows.append(features)
        
        df = pd.DataFrame(rows)
        
        print(f"  - Prepared {len(df)} training samples")
        print(f"  - Features: {len([c for c in df.columns if c not in ['label_click', 'label_dwell', 'user_id', 'article_id']])}")
        
        return df
