"""
Two-Tower Model: Joint training of user and item encoders
"""
import numpy as np
import pickle
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm

from src.item_encoder import ItemEncoder
from src.user_encoder import UserEncoder


class TwoTowerModel:
    """Two-tower architecture for candidate generation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.item_encoder = ItemEncoder(config['item_encoder'])
        self.user_encoder = UserEncoder(config['user_encoder'])
        
        self.article_metadata = {}
        self.item_embeddings = {}
        
    def fit_item_encoder(self, article_metadata: Dict[str, Dict]):
        """
        Fit item encoder on article corpus
        
        Args:
            article_metadata: {article_id: {title, category, ...}}
        """
        print("Fitting item encoder...")
        self.article_metadata = article_metadata
        self.item_encoder.fit(article_metadata)
        
        # Precompute all item embeddings
        print("Precomputing item embeddings...")
        self.item_embeddings = self.item_encoder.encode_batch(article_metadata)
        
        print(f"  - Encoded {len(self.item_embeddings)} articles")
    
    def get_user_embedding(self,
                          user_id: str,
                          clicked_article_ids: List[str],
                          session_state: Dict) -> np.ndarray:
        """
        Get user embedding for a given session
        
        Args:
            user_id: User identifier
            clicked_article_ids: List of articles user clicked in session
            session_state: Session features
        
        Returns: user embedding
        """
        # Get embeddings of clicked articles
        clicked_embeddings = [
            self.item_embeddings[aid]
            for aid in clicked_article_ids
            if aid in self.item_embeddings
        ]
        
        # Encode user
        user_embedding = self.user_encoder.encode_user(
            clicked_embeddings,
            session_state
        )
        
        return user_embedding
    
    def save(self, save_dir: str):
        """Save model components"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save encoders
        with open(save_path / 'item_encoder.pkl', 'wb') as f:
            pickle.dump(self.item_encoder, f)
        
        with open(save_path / 'user_encoder.pkl', 'wb') as f:
            pickle.dump(self.user_encoder, f)
        
        # Save precomputed embeddings
        with open(save_path / 'item_embeddings.pkl', 'wb') as f:
            pickle.dump(self.item_embeddings, f)
        
        with open(save_path / 'article_metadata.pkl', 'wb') as f:
            pickle.dump(self.article_metadata, f)
        
        print(f"Model saved to {save_path}")
    
    def load(self, save_dir: str):
        """Load model components"""
        save_path = Path(save_dir)
        
        with open(save_path / 'item_encoder.pkl', 'rb') as f:
            self.item_encoder = pickle.load(f)
        
        with open(save_path / 'user_encoder.pkl', 'rb') as f:
            self.user_encoder = pickle.load(f)
        
        with open(save_path / 'item_embeddings.pkl', 'rb') as f:
            self.item_embeddings = pickle.load(f)
        
        with open(save_path / 'article_metadata.pkl', 'rb') as f:
            self.article_metadata = pickle.load(f)
        
        print(f"Model loaded from {save_path}")
