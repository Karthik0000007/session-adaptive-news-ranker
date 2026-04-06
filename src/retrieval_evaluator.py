"""
Retrieval Evaluation: Measure candidate generation quality
"""
import numpy as np
from typing import Dict, List, Set
from collections import defaultdict


class RetrievalEvaluator:
    """Evaluate retrieval system performance"""
    
    @staticmethod
    def recall_at_k(clicked_items: Set[str], 
                   retrieved_items: List[str], 
                   k: int) -> float:
        """
        Recall@K: % of clicked items in top-k retrieved
        
        recall@k = |clicked ∩ retrieved[:k]| / |clicked|
        """
        if not clicked_items:
            return 0.0
        
        retrieved_k = set(retrieved_items[:k])
        hits = len(clicked_items & retrieved_k)
        
        return hits / len(clicked_items)
    
    @staticmethod
    def hit_rate_at_k(clicked_items: Set[str],
                     retrieved_items: List[str],
                     k: int) -> float:
        """
        Hit Rate@K: Did we retrieve ANY clicked item in top-k?
        
        hit_rate@k = 1 if |clicked ∩ retrieved[:k]| > 0 else 0
        """
        retrieved_k = set(retrieved_items[:k])
        return 1.0 if (clicked_items & retrieved_k) else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(clicked_items: Set[str],
                            retrieved_items: List[str]) -> float:
        """
        MRR: Average rank of first clicked item
        
        mrr = 1 / rank_of_first_hit
        """
        for rank, item in enumerate(retrieved_items, 1):
            if item in clicked_items:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def evaluate_batch(sessions: Dict[str, Dict],
                      retrieval_results: Dict[str, List[str]],
                      k_values: List[int] = [10, 50, 100]) -> Dict:
        """
        Evaluate retrieval on batch of sessions
        
        Args:
            sessions: {session_id: {clicked_items: [...]}}
            retrieval_results: {session_id: [retrieved_articles]}
            k_values: K values to evaluate at
        
        Returns: Metrics dict
        """
        metrics = defaultdict(list)
        
        for session_id, session_data in sessions.items():
            clicked_items = set(session_data.get('clicked_items', []))
            retrieved = retrieval_results.get(session_id, [])
            
            if not clicked_items or not retrieved:
                continue
            
            # Recall@K
            for k in k_values:
                recall = RetrievalEvaluator.recall_at_k(clicked_items, retrieved, k)
                metrics[f'recall@{k}'].append(recall)
            
            # Hit Rate@K
            for k in k_values:
                hit_rate = RetrievalEvaluator.hit_rate_at_k(clicked_items, retrieved, k)
                metrics[f'hit_rate@{k}'].append(hit_rate)
            
            # MRR
            mrr = RetrievalEvaluator.mean_reciprocal_rank(clicked_items, retrieved)
            metrics['mrr'].append(mrr)
        
        # Average metrics
        results = {}
        for metric_name, values in metrics.items():
            if values:
                results[metric_name] = np.mean(values)
                results[f'{metric_name}_std'] = np.std(values)
        
        return results
    
    @staticmethod
    def print_metrics(metrics: Dict):
        """Pretty print evaluation metrics"""
        print("\n" + "=" * 60)
        print("Retrieval Evaluation Results")
        print("=" * 60)
        
        for metric_name in sorted(metrics.keys()):
            if '_std' not in metric_name:
                value = metrics[metric_name]
                std = metrics.get(f'{metric_name}_std', 0)
                print(f"{metric_name:20s}: {value:.4f} (±{std:.4f})")
        
        print("=" * 60)
