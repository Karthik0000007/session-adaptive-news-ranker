"""
Decision Layer: Integrates multi-objective scoring with adaptive weighting
"""
import numpy as np
from typing import Dict, List, Tuple

from src.objective_scorer import ObjectiveScorer
from src.weight_adapter import WeightAdapter, WeightStrategy


class DecisionLayer:
    """
    Decision-making layer that combines:
    - Multi-objective scoring (Phase 4)
    - Adaptive weighting (Phase 5)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.scorer = ObjectiveScorer(config.get('scoring', {}))
        self.weight_adapter = WeightAdapter(config.get('weighting', {}))
        
        # Strategy
        self.strategy = config.get('weight_strategy', WeightStrategy.FIXED)
    
    def rank_candidates(self,
                       candidates: List[Dict],
                       session_state: Dict,
                       item_embeddings: Dict[str, np.ndarray],
                       item_stats: Dict[str, Dict],
                       article_metadata: Dict[str, Dict]) -> List[Tuple[str, float, Dict]]:
        """
        Rank candidates using multi-objective scoring
        
        Args:
            candidates: List of {article_id, ctr_score, dwell_score, retention_score}
            session_state: Current session features
            item_embeddings: Article embeddings for diversity
            item_stats: Item statistics for novelty
            article_metadata: Article metadata
        
        Returns: Ranked list of (article_id, score, component_scores)
        """
        # Get weights based on strategy
        if self.strategy == WeightStrategy.FIXED:
            weights = self.weight_adapter.get_baseline_weights()
        elif self.strategy == WeightStrategy.RULE_BASED:
            weights = self.weight_adapter.get_weights(session_state)
        else:
            # Bandit strategy (Phase 6)
            weights = self.weight_adapter.get_baseline_weights()
        
        # Score and rank candidates
        ranked_results = self.scorer.score_candidates(
            candidates=candidates,
            weights=weights,
            item_embeddings=item_embeddings,
            item_stats=item_stats,
            article_metadata=article_metadata
        )
        
        return ranked_results
    
    def get_current_weights(self, session_state: Dict) -> List[float]:
        """Get current weights for given session state"""
        if self.strategy == WeightStrategy.FIXED:
            return self.weight_adapter.get_baseline_weights()
        elif self.strategy == WeightStrategy.RULE_BASED:
            return self.weight_adapter.get_weights(session_state, smooth=False)
        else:
            return self.weight_adapter.get_baseline_weights()
    
    def get_weight_explanation(self, session_state: Dict) -> str:
        """Get explanation of weight selection"""
        if self.strategy == WeightStrategy.FIXED:
            return "Fixed baseline weights"
        elif self.strategy == WeightStrategy.RULE_BASED:
            return self.weight_adapter.get_rule_explanation(session_state)
        else:
            return "Unknown strategy"
    
    def set_strategy(self, strategy: str):
        """Change weight strategy"""
        self.strategy = strategy
        if strategy == WeightStrategy.RULE_BASED:
            self.weight_adapter.reset()
    
    def reset_session(self):
        """Reset session-specific state"""
        self.weight_adapter.reset()


class RankingPipeline:
    """
    End-to-end ranking pipeline:
    Retrieval → Base Ranking → Decision Layer
    """
    
    def __init__(self, 
                 retrieval_system,
                 ranking_system,
                 decision_layer: DecisionLayer):
        self.retrieval_system = retrieval_system
        self.ranking_system = ranking_system
        self.decision_layer = decision_layer
    
    def rank(self,
            user_id: str,
            clicked_article_ids: List[str],
            session_state: Dict,
            article_metadata: Dict[str, Dict],
            k: int = 20) -> List[Tuple[str, float, Dict]]:
        """
        Full ranking pipeline
        
        Args:
            user_id: User identifier
            clicked_article_ids: Articles clicked in session
            session_state: Session features
            article_metadata: Article metadata
            k: Number of items to return
        
        Returns: Top-k ranked articles with scores
        """
        # Step 1: Retrieve candidates
        candidate_ids = self.retrieval_system.retrieve(
            user_id=user_id,
            clicked_article_ids=clicked_article_ids,
            session_state=session_state,
            k=100
        )
        
        # Step 2: Get base ranking scores
        candidates = []
        for article_id in candidate_ids:
            scores = self.ranking_system.predict(
                user_id=user_id,
                article_id=article_id,
                session_state=session_state,
                article_metadata=article_metadata,
                clicked_history=clicked_article_ids
            )
            
            candidates.append({
                'article_id': article_id,
                'ctr_score': scores['ctr_score'],
                'dwell_score': scores['dwell_score'],
                'retention_score': scores['retention_score']
            })
        
        # Step 3: Decision layer (multi-objective + adaptive weights)
        item_embeddings = self.retrieval_system.model.item_embeddings
        item_stats = self.ranking_system.feature_builder.item_stats
        
        ranked_results = self.decision_layer.rank_candidates(
            candidates=candidates,
            session_state=session_state,
            item_embeddings=item_embeddings,
            item_stats=item_stats,
            article_metadata=article_metadata
        )
        
        # Return top-k
        return ranked_results[:k]
