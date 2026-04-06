"""
Negative sampling for training data generation
"""
import random
from typing import List, Dict, Set


class NegativeSampler:
    """Generate negative samples for ranking training"""
    
    def __init__(self, num_negatives: int = 10):
        self.num_negatives = num_negatives
        random.seed(42)
    
    def sample_negatives(self, 
                        positive_article: str,
                        candidate_pool: List[str],
                        exclude_set: Set[str]) -> List[str]:
        """
        Sample negative articles for a positive interaction
        
        Args:
            positive_article: The clicked article
            candidate_pool: All available articles
            exclude_set: Articles to exclude (already seen in session)
        
        Returns:
            List of negative article IDs
        """
        # Filter candidates
        valid_candidates = [
            article for article in candidate_pool 
            if article not in exclude_set and article != positive_article
        ]
        
        # Sample
        num_to_sample = min(self.num_negatives, len(valid_candidates))
        negatives = random.sample(valid_candidates, num_to_sample)
        
        return negatives
    
    def create_candidate_set(self,
                            positive_article: str,
                            negatives: List[str]) -> List[Dict]:
        """
        Create candidate set with positive and negatives
        
        Returns list with labels
        """
        candidates = []
        
        # Add positive
        candidates.append({
            'article_id': positive_article,
            'label': 1
        })
        
        # Add negatives
        for neg in negatives:
            candidates.append({
                'article_id': neg,
                'label': 0
            })
        
        return candidates
