"""
User Tower: Session-Aware User Encoder
Encodes user session state into embeddings
"""
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import normalize


class UserEncoder:
    """Encode user session state into embeddings"""
    
    def __init__(self, config: Dict):
        self.embedding_dim = config['embedding_dim']
        self.history_weight = config['history_weight']
        self.session_weight = config['session_weight']
        
        # Session feature normalization bounds
        self.session_feature_bounds = {
            'session_length': (1, 50),
            'avg_dwell_time': (0, 300),
            'click_rate': (0, 1),
            'skip_rate': (0, 1),
            'click_entropy': (0, 3),
            'fatigue_score': (0, 1)
        }
        
        np.random.seed(42)
    
    def encode_user(self,
                   clicked_embeddings: List[np.ndarray],
                   session_state: Dict) -> np.ndarray:
        """
        Encode user based on clicked articles and session state
        
        Args:
            clicked_embeddings: List of item embeddings user clicked
            session_state: Session features dict
        
        Returns: user embedding of shape (embedding_dim,)
        """
        # History embedding: mean of clicked items
        if clicked_embeddings:
            history_vec = np.mean(clicked_embeddings, axis=0)
            history_vec = normalize(history_vec.reshape(1, -1))[0]
        else:
            history_vec = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Session context embedding
        session_vec = self._encode_session_features(session_state)
        
        # Combine
        user_embedding = (
            self.history_weight * history_vec +
            self.session_weight * session_vec
        )
        
        # Normalize
        user_embedding = normalize(user_embedding.reshape(1, -1))[0]
        
        return user_embedding.astype(np.float32)
    
    def _encode_session_features(self, session_state: Dict) -> np.ndarray:
        """
        Encode session features into embedding
        
        Simple approach: normalize features and project to embedding space
        """
        # Extract and normalize features
        features = []
        
        for key in ['session_length', 'avg_dwell_time', 'click_rate', 
                    'skip_rate', 'click_entropy', 'fatigue_score']:
            value = session_state.get(key, 0)
            
            # Normalize to [0, 1]
            min_val, max_val = self.session_feature_bounds[key]
            normalized = (value - min_val) / (max_val - min_val + 1e-8)
            normalized = np.clip(normalized, 0, 1)
            
            features.append(normalized)
        
        features = np.array(features, dtype=np.float32)
        
        # Project to embedding dimension using random projection
        # (In production, this would be learned)
        projection = np.random.randn(len(features), self.embedding_dim)
        projection = normalize(projection, axis=0)
        
        session_vec = features @ projection
        session_vec = normalize(session_vec.reshape(1, -1))[0]
        
        return session_vec.astype(np.float32)
    
    def encode_batch(self,
                    user_sessions: Dict[str, List[np.ndarray]],
                    session_states: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """
        Encode multiple users
        
        Args:
            user_sessions: {user_id: [clicked_embeddings]}
            session_states: {user_id: session_state_dict}
        
        Returns: {user_id: embedding}
        """
        embeddings = {}
        
        for user_id, clicked_embs in user_sessions.items():
            session_state = session_states.get(user_id, {})
            embeddings[user_id] = self.encode_user(clicked_embs, session_state)
        
        return embeddings
