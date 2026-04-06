"""
FAISS Index: Fast similarity search for candidate retrieval
"""
import numpy as np
import faiss
import pickle
from typing import Dict, List, Tuple
from pathlib import Path


class FAISSIndex:
    """FAISS-based vector search for efficient retrieval"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = None
        self.article_ids = []
        self.id_to_article = {}
        
    def build(self, item_embeddings: Dict[str, np.ndarray]):
        """
        Build FAISS index from item embeddings
        
        Args:
            item_embeddings: {article_id: embedding}
        """
        print("Building FAISS index...")
        
        # Prepare embeddings matrix
        article_ids = list(item_embeddings.keys())
        embeddings = np.array([
            item_embeddings[aid] for aid in article_ids
        ], dtype=np.float32)
        
        # Create index (using inner product for dot product similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
        self.article_ids = article_ids
        self.id_to_article = {i: aid for i, aid in enumerate(article_ids)}
        
        print(f"  - Index built with {len(article_ids)} articles")
        print(f"  - Embedding dimension: {self.embedding_dim}")
    
    def search(self, user_embedding: np.ndarray, k: int = 100) -> List[str]:
        """
        Retrieve top-k candidate articles for user
        
        Args:
            user_embedding: User embedding vector
            k: Number of candidates to retrieve
        
        Returns: List of top-k article IDs
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Ensure embedding is float32 and 2D
        user_embedding = np.array(user_embedding, dtype=np.float32).reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(user_embedding, k)
        
        # Convert indices to article IDs
        candidate_ids = [self.id_to_article[idx] for idx in indices[0]]
        
        return candidate_ids
    
    def search_batch(self, user_embeddings: np.ndarray, k: int = 100) -> List[List[str]]:
        """
        Retrieve candidates for multiple users
        
        Args:
            user_embeddings: Array of shape (num_users, embedding_dim)
            k: Number of candidates per user
        
        Returns: List of candidate lists
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        user_embeddings = np.array(user_embeddings, dtype=np.float32)
        distances, indices = self.index.search(user_embeddings, k)
        
        results = []
        for idx_list in indices:
            candidates = [self.id_to_article[idx] for idx in idx_list]
            results.append(candidates)
        
        return results
    
    def save(self, save_dir: str):
        """Save index to disk"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / 'faiss.index'))
        
        # Save metadata
        metadata = {
            'article_ids': self.article_ids,
            'id_to_article': self.id_to_article,
            'embedding_dim': self.embedding_dim
        }
        
        with open(save_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Index saved to {save_path}")
    
    def load(self, save_dir: str):
        """Load index from disk"""
        save_path = Path(save_dir)
        
        # Load FAISS index
        self.index = faiss.read_index(str(save_path / 'faiss.index'))
        
        # Load metadata
        with open(save_path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.article_ids = metadata['article_ids']
        self.id_to_article = metadata['id_to_article']
        self.embedding_dim = metadata['embedding_dim']
        
        print(f"Index loaded from {save_path}")
