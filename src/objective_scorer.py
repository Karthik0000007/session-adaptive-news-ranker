"""
Multi-Objective Scoring: Combines ranking signals into final scores
"""
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler


class ObjectiveScorer:
    """Compute and combine multiple objective scores"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.beta = config.get('novelty_beta', 0.7)
        self.lambda_decay = config.get('freshness_decay', 0.1)
        
    def compute_engagement_score(self, 
                                 ctr_score: float, 
                                 dwell_score: float,
                                 alpha: float = 0.6) -> float:
        """
        Engagement = alpha * CTR + (1 - alpha) * Dwell
        
        Both inputs should be in [0, 1]
        """
        return alpha * ctr_score + (1 - alpha) * dwell_score
    
    def compute_retention_score(self, retention_score: float) -> float:
        """
        Retention proxy from model
        
        Already in [0, 1] from calibrated model
        """
        return retention_score
    
    def compute_diversity_score(self,
                                candidate_embedding: np.ndarray,
                                ranked_embeddings: List[np.ndarray]) -> float:
        """
        Diversity = 1 - avg_similarity(candidate, ranked_list)
        
        Higher score = more diverse from existing items
        """
        if not ranked_embeddings:
            return 1.0  # First item is maximally diverse
        
        # Compute cosine similarity to all ranked items
        similarities = []
        for ranked_emb in ranked_embeddings:
            sim = self._cosine_similarity(candidate_embedding, ranked_emb)
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        
        return max(0.0, diversity)  # Ensure non-negative
    
    def compute_novelty_score(self,
                             article_id: str,
                             item_stats: Dict[str, Dict],
                             article_metadata: Dict[str, Dict]) -> float:
        """
        Novelty = beta * freshness + (1 - beta) * unpopularity
        
        Encourages fresh and less-popular content
        """
        stats = item_stats.get(article_id, {})
        
        # Freshness (simplified - would use actual timestamp in production)
        # For now, use inverse of impressions as proxy
        impressions = stats.get('impressions', 1)
        freshness = np.exp(-self.lambda_decay * np.log1p(impressions))
        
        # Unpopularity
        popularity = stats.get('popularity', 0)
        unpopularity = 1.0 / (1.0 + np.log1p(popularity))
        
        novelty = self.beta * freshness + (1 - self.beta) * unpopularity
        
        return novelty
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def normalize_scores(self, scores: List[float]) -> np.ndarray:
        """
        Normalize scores to [0, 1] using min-max scaling
        
        Critical for fair weighting across objectives
        """
        scores_array = np.array(scores).reshape(-1, 1)
        
        if len(scores) == 1:
            return np.array([1.0])
        
        min_val = scores_array.min()
        max_val = scores_array.max()
        
        if max_val - min_val < 1e-8:
            return np.ones(len(scores))
        
        normalized = (scores_array - min_val) / (max_val - min_val + 1e-8)
        
        return normalized.flatten()
    
    def compute_final_score(self,
                           engagement: float,
                           retention: float,
                           diversity: float,
                           novelty: float,
                           weights: List[float]) -> float:
        """
        Final score = w1*E + w2*R + w3*D + w4*N
        
        All inputs should be normalized to [0, 1]
        Weights should sum to 1
        """
        w1, w2, w3, w4 = weights
        
        score = (
            w1 * engagement +
            w2 * retention +
            w3 * diversity +
            w4 * novelty
        )
        
        return score
    
    def score_candidates(self,
                        candidates: List[Dict],
                        weights: List[float],
                        item_embeddings: Dict[str, np.ndarray],
                        item_stats: Dict[str, Dict],
                        article_metadata: Dict[str, Dict]) -> List[Tuple[str, float, Dict]]:
        """
        Score all candidates with given weights
        
        Returns: List of (article_id, final_score, component_scores)
        """
        # Collect all objective scores
        engagement_scores = []
        retention_scores = []
        novelty_scores = []
        
        for candidate in candidates:
            # Engagement
            eng = self.compute_engagement_score(
                candidate['ctr_score'],
                candidate['dwell_score']
            )
            engagement_scores.append(eng)
            
            # Retention
            ret = self.compute_retention_score(candidate['retention_score'])
            retention_scores.append(ret)
            
            # Novelty
            nov = self.compute_novelty_score(
                candidate['article_id'],
                item_stats,
                article_metadata
            )
            novelty_scores.append(nov)
        
        # Normalize scores across candidates
        engagement_norm = self.normalize_scores(engagement_scores)
        retention_norm = self.normalize_scores(retention_scores)
        novelty_norm = self.normalize_scores(novelty_scores)
        
        # Greedy reranking with diversity
        ranked_results = []
        ranked_embeddings = []
        remaining_indices = list(range(len(candidates)))
        
        while remaining_indices:
            best_idx = None
            best_score = -1
            best_components = None
            
            for idx in remaining_indices:
                candidate = candidates[idx]
                article_id = candidate['article_id']
                
                # Compute diversity
                candidate_emb = item_embeddings.get(article_id)
                if candidate_emb is not None:
                    diversity = self.compute_diversity_score(
                        candidate_emb,
                        ranked_embeddings
                    )
                else:
                    diversity = 0.5  # Default if embedding missing
                
                # Compute final score
                final_score = self.compute_final_score(
                    engagement_norm[idx],
                    retention_norm[idx],
                    diversity,
                    novelty_norm[idx],
                    weights
                )
                
                if final_score > best_score:
                    best_score = final_score
                    best_idx = idx
                    best_components = {
                        'engagement': float(engagement_norm[idx]),
                        'retention': float(retention_norm[idx]),
                        'diversity': float(diversity),
                        'novelty': float(novelty_norm[idx])
                    }
            
            # Add best candidate to ranked list
            best_candidate = candidates[best_idx]
            ranked_results.append((
                best_candidate['article_id'],
                best_score,
                best_components
            ))
            
            # Update ranked embeddings
            best_emb = item_embeddings.get(best_candidate['article_id'])
            if best_emb is not None:
                ranked_embeddings.append(best_emb)
            
            remaining_indices.remove(best_idx)
        
        return ranked_results
