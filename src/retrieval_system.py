"""
Retrieval System: Complete candidate generation pipeline
"""
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from src.two_tower_model import TwoTowerModel
from src.faiss_index import FAISSIndex


class RetrievalSystem:
    """End-to-end candidate retrieval system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = TwoTowerModel(config['two_tower'])
        self.index = FAISSIndex(config['two_tower']['item_encoder']['embedding_dim'])
        
    def train(self, article_metadata: Dict[str, Dict]):
        """
        Train retrieval system
        
        Args:
            article_metadata: {article_id: {title, category, ...}}
        """
        print("=" * 60)
        print("Training Retrieval System")
        print("=" * 60)
        
        # Fit item encoder and precompute embeddings
        self.model.fit_item_encoder(article_metadata)
        
        # Build FAISS index
        print("\nBuilding FAISS index...")
        self.index.build(self.model.item_embeddings)
        
        print("\n" + "=" * 60)
        print("Retrieval System Training Complete")
        print("=" * 60)
    
    def retrieve(self,
                user_id: str,
                clicked_article_ids: List[str],
                session_state: Dict,
                k: int = 100) -> List[str]:
        """
        Retrieve top-k candidate articles for user
        
        Args:
            user_id: User identifier
            clicked_article_ids: Articles user clicked in session
            session_state: Session features
            k: Number of candidates
        
        Returns: List of top-k article IDs
        """
        # Get user embedding
        user_embedding = self.model.get_user_embedding(
            user_id,
            clicked_article_ids,
            session_state
        )
        
        # Search index
        candidates = self.index.search(user_embedding, k=k)
        
        return candidates
    
    def retrieve_batch(self,
                      user_data: List[Dict],
                      k: int = 100) -> List[List[str]]:
        """
        Retrieve candidates for multiple users
        
        Args:
            user_data: List of {user_id, clicked_ids, session_state}
            k: Number of candidates per user
        
        Returns: List of candidate lists
        """
        # Get user embeddings
        user_embeddings = []
        
        for data in user_data:
            user_emb = self.model.get_user_embedding(
                data['user_id'],
                data['clicked_ids'],
                data['session_state']
            )
            user_embeddings.append(user_emb)
        
        user_embeddings = np.array(user_embeddings, dtype=np.float32)
        
        # Batch search
        candidates = self.index.search_batch(user_embeddings, k=k)
        
        return candidates
    
    def save(self, save_dir: str):
        """Save retrieval system"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(save_path / 'model'))
        self.index.save(str(save_path / 'index'))
        
        print(f"Retrieval system saved to {save_path}")
    
    def load(self, save_dir: str):
        """Load retrieval system"""
        save_path = Path(save_dir)
        
        self.model.load(str(save_path / 'model'))
        self.index.load(str(save_path / 'index'))
        
        print(f"Retrieval system loaded from {save_path}")
