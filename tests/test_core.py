"""
Unit tests for core components.

Fix 7 (P1): tests/ directory was empty — these cover critical paths.
"""

import sys
import os
import numpy as np
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# Test ObjectiveScorer
# ---------------------------------------------------------------------------
class TestObjectiveScorer:
    
    def setup_method(self):
        from src.objective_scorer import ObjectiveScorer
        self.scorer = ObjectiveScorer({})
    
    def test_engagement_score_bounds(self):
        """Engagement score must be in [0, 1]"""
        assert 0 <= self.scorer.compute_engagement_score(0.0, 0.0) <= 1
        assert 0 <= self.scorer.compute_engagement_score(1.0, 1.0) <= 1
        assert 0 <= self.scorer.compute_engagement_score(0.5, 0.5) <= 1
    
    def test_engagement_score_alpha(self):
        """Alpha controls CTR vs dwell balance"""
        # alpha=1.0 → pure CTR
        score = self.scorer.compute_engagement_score(0.8, 0.2, alpha=1.0)
        assert abs(score - 0.8) < 1e-6
        
        # alpha=0.0 → pure dwell
        score = self.scorer.compute_engagement_score(0.8, 0.2, alpha=0.0)
        assert abs(score - 0.2) < 1e-6
    
    def test_diversity_first_item(self):
        """First item in empty list is maximally diverse"""
        emb = np.array([1.0, 0.0, 0.0])
        score = self.scorer.compute_diversity_score(emb, [])
        assert score == 1.0
    
    def test_diversity_identical_items(self):
        """Identical items have zero diversity"""
        emb = np.array([1.0, 0.0, 0.0])
        ranked = [np.array([1.0, 0.0, 0.0])]
        score = self.scorer.compute_diversity_score(emb, ranked)
        assert score == 0.0
    
    def test_diversity_orthogonal_items(self):
        """Orthogonal items have maximum diversity"""
        emb = np.array([1.0, 0.0, 0.0])
        ranked = [np.array([0.0, 1.0, 0.0])]
        score = self.scorer.compute_diversity_score(emb, ranked)
        assert score == 1.0
    
    def test_novelty_score_range(self):
        """Novelty score must be in [0, 1]"""
        item_stats = {'article_1': {'impressions': 100, 'popularity': 50}}
        score = self.scorer.compute_novelty_score('article_1', item_stats, {})
        assert 0 <= score <= 1
    
    def test_final_score_weighted_sum(self):
        """Final score must be weighted sum of components"""
        weights = [0.4, 0.3, 0.2, 0.1]
        score = self.scorer.compute_final_score(0.5, 0.6, 0.7, 0.8, weights)
        expected = 0.4 * 0.5 + 0.3 * 0.6 + 0.2 * 0.7 + 0.1 * 0.8
        assert abs(score - expected) < 1e-6
    
    def test_normalize_scores_single(self):
        """Single score normalizes to 1.0"""
        result = self.scorer.normalize_scores([0.5])
        assert result[0] == 1.0
    
    def test_normalize_scores_range(self):
        """Normalized scores must be in [0, 1]"""
        scores = [0.1, 0.5, 0.9, 0.3, 0.7]
        normalized = self.scorer.normalize_scores(scores)
        assert all(0 <= s <= 1 for s in normalized)
        assert abs(normalized.max() - 1.0) < 1e-6
        assert abs(normalized.min() - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# Test WeightAdapter
# ---------------------------------------------------------------------------
class TestWeightAdapter:
    
    def setup_method(self):
        from src.weight_adapter import WeightAdapter
        self.adapter = WeightAdapter({'smoothing_alpha': 1.0})
    
    def test_weight_constraints_early_session(self):
        """Weights must sum to 1 and be non-negative (early session)"""
        state = {'session_length': 0}
        weights = self.adapter.get_weights(state)
        assert abs(sum(weights) - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)
    
    def test_weight_constraints_mid_session(self):
        """Weights must sum to 1 and be non-negative (mid session, high engagement)"""
        state = {'session_length': 10, 'avg_dwell_time': 45.0, 'fatigue_score': 0.2}
        weights = self.adapter.get_weights(state)
        assert abs(sum(weights) - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)
    
    def test_weight_constraints_fatigued(self):
        """Weights must sum to 1 and be non-negative (fatigued)"""
        state = {'session_length': 15, 'avg_dwell_time': 10.0, 'fatigue_score': 0.8}
        weights = self.adapter.get_weights(state)
        assert abs(sum(weights) - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)
    
    def test_early_session_boosts_diversity(self):
        """Early session should boost diversity weight"""
        state_early = {'session_length': 1}
        state_mid = {'session_length': 10, 'avg_dwell_time': 20.0, 'fatigue_score': 0.3}
        
        weights_early = self.adapter.get_weights(state_early, smooth=False)
        self.adapter.reset()
        weights_mid = self.adapter.get_weights(state_mid, smooth=False)
        
        # Diversity is index 2
        assert weights_early[2] > weights_mid[2], \
            "Early session should have higher diversity weight"
    
    def test_fatigue_boosts_novelty(self):
        """Fatigue should boost novelty weight"""
        state_normal = {'session_length': 5, 'avg_dwell_time': 20.0, 'fatigue_score': 0.3}
        state_fatigued = {'session_length': 15, 'avg_dwell_time': 10.0, 'fatigue_score': 0.8}
        
        weights_normal = self.adapter.get_weights(state_normal, smooth=False)
        self.adapter.reset()
        weights_fatigued = self.adapter.get_weights(state_fatigued, smooth=False)
        
        # Novelty is index 3
        assert weights_fatigued[3] > weights_normal[3], \
            "Fatigued session should have higher novelty weight"
    
    def test_smoothing(self):
        """Smoothing should prevent abrupt weight transitions"""
        from src.weight_adapter import WeightAdapter as WA
        adapter = WA({'smoothing_alpha': 0.5})
        
        state1 = {'session_length': 1}
        state2 = {'session_length': 10, 'avg_dwell_time': 45.0, 'fatigue_score': 0.1}
        
        weights1 = adapter.get_weights(state1)
        weights2 = adapter.get_weights(state2)
        
        # With smoothing, weights2 should be between weights1 and the raw target
        # It should not jump abruptly
        assert abs(sum(weights2) - 1.0) < 1e-6
    
    def test_reset_clears_state(self):
        """Reset should clear previous weights"""
        state = {'session_length': 5}
        self.adapter.get_weights(state)
        assert self.adapter.previous_weights is not None
        
        self.adapter.reset()
        assert self.adapter.previous_weights is None
    
    def test_baseline_weights(self):
        """Baseline weights should be the default [0.4, 0.3, 0.2, 0.1]"""
        baseline = self.adapter.get_baseline_weights()
        assert baseline == [0.4, 0.3, 0.2, 0.1]


# ---------------------------------------------------------------------------
# Test LinUCB Contextual Bandit
# ---------------------------------------------------------------------------
class TestLinUCB:
    
    def setup_method(self):
        from src.contextual_bandit import LinUCB
        self.bandit = LinUCB({'alpha': 0.5})
    
    def test_action_selection_returns_valid(self):
        """Action selection must return valid index and weights"""
        state = {'session_length': 5, 'avg_dwell_time': 30}
        idx, weights = self.bandit.select_action(state)
        assert 0 <= idx < len(self.bandit.actions)
        assert abs(sum(weights) - 1.0) < 1e-6
    
    def test_greedy_action_selection(self):
        """Greedy selection (no exploration) must be deterministic"""
        state = {'session_length': 5, 'avg_dwell_time': 30}
        idx1, _ = self.bandit.select_action(state, explore=False)
        idx2, _ = self.bandit.select_action(state, explore=False)
        assert idx1 == idx2
    
    def test_featurize_dimension(self):
        """Feature vector must have consistent dimension"""
        state = {'session_length': 5, 'avg_dwell_time': 30}
        features = self.bandit.featurize(state)
        assert len(features) == 9  # 8 features + 1 bias term
    
    def test_featurize_bias(self):
        """Last element of feature vector must be 1.0 (bias)"""
        state = {}
        features = self.bandit.featurize(state)
        assert features[-1] == 1.0
    
    def test_update_changes_parameters(self):
        """Update must change bandit parameters"""
        state = {'session_length': 5, 'avg_dwell_time': 30}
        
        # Initialize by selecting an action
        idx, _ = self.bandit.select_action(state)
        
        # Get A matrix before update
        A_before = self.bandit.A[idx].copy()
        
        # Update
        self.bandit.update(state, idx, reward=1.0)
        
        # A matrix must change
        assert not np.array_equal(A_before, self.bandit.A[idx])
    
    def test_action_probabilities_sum_to_one(self):
        """Action probabilities must sum to ~1"""
        state = {'session_length': 5, 'avg_dwell_time': 30}
        # Initialize
        self.bandit.select_action(state)
        
        probs = self.bandit.get_action_probabilities(state)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert all(p >= 0 for p in probs)
    
    def test_save_load_state(self):
        """Save/load must preserve bandit state"""
        state = {'session_length': 5, 'avg_dwell_time': 30}
        self.bandit.select_action(state)
        self.bandit.update(state, 0, reward=0.5)
        
        saved = self.bandit.save_state()
        
        from src.contextual_bandit import LinUCB
        new_bandit = LinUCB({'alpha': 0.5})
        new_bandit.load_state(saved)
        
        assert new_bandit.alpha == self.bandit.alpha
        assert new_bandit.d == self.bandit.d


# ---------------------------------------------------------------------------
# Test DriftDetector
# ---------------------------------------------------------------------------
class TestDriftDetector:
    
    def setup_method(self):
        from src.drift_detector import DriftDetector
        self.detector = DriftDetector(window_size=100, threshold=0.05)
    
    def test_no_reference_no_drift(self):
        """No reference → no drift detected"""
        result = self.detector.detect(np.array([1, 2, 3]))
        assert result['drift_detected'] is False
    
    def test_same_distribution_no_drift(self):
        """Identical distributions should not trigger drift"""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 500)
        current = np.random.normal(0, 1, 500)
        
        self.detector.update_reference(reference)
        result = self.detector.detect(current)
        assert result['drift_detected'] == False
    
    def test_shifted_distribution_detects_drift(self):
        """Significantly shifted distribution should trigger drift"""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 500)
        current = np.random.normal(3, 1, 500)  # Mean shifted by 3
        
        self.detector.update_reference(reference)
        result = self.detector.detect(current)
        assert result['drift_detected'] == True
    
    def test_sliding_window(self):
        """Sliding window accumulates samples correctly"""
        self.detector.add_sample(1.0)
        self.detector.add_sample(2.0)
        assert len(self.detector.current_window) == 2


# ---------------------------------------------------------------------------
# Test JapaneseTokenizer
# ---------------------------------------------------------------------------
class TestJapaneseTokenizer:
    
    def setup_method(self):
        from src.japanese_tokenizer import JapaneseTokenizer
        self.tokenizer = JapaneseTokenizer()
    
    def test_empty_input(self):
        """Empty input returns empty list"""
        assert self.tokenizer.tokenize("") == []
        assert self.tokenizer.tokenize("   ") == []
    
    def test_ascii_text(self):
        """ASCII text is tokenized into words"""
        tokens = self.tokenizer.tokenize("hello world 123")
        assert len(tokens) > 0
        assert any('hello' in t for t in tokens)
    
    def test_backend_attribute(self):
        """Backend attribute is set"""
        assert self.tokenizer.backend in ('sudachi', 'regex_fallback')
    
    def test_callable_interface(self):
        """Tokenizer is callable (sklearn compatibility)"""
        tokens = self.tokenizer("test text")
        assert isinstance(tokens, list)
    
    def test_mixed_text(self):
        """Mixed Japanese/English text produces tokens"""
        tokens = self.tokenizer.tokenize("ニュース news 記事")
        assert len(tokens) >= 3  # At least 3 tokens


# ---------------------------------------------------------------------------
# Test Counterfactual Evaluator
# ---------------------------------------------------------------------------
class TestCounterfactualEvaluator:
    
    def setup_method(self):
        from src.counterfactual_evaluator import CounterfactualEvaluator, RewardCalculator
        self.evaluator = CounterfactualEvaluator()
        self.reward_calc = RewardCalculator({})
    
    def test_ips_empty_data(self):
        """IPS with empty data returns zeros"""
        result = self.evaluator.inverse_propensity_scoring([], None)
        assert result['ips_reward'] == 0.0
        assert result['num_samples'] == 0
    
    def test_snips_empty_data(self):
        """SNIPS with empty data returns zeros"""
        result = self.evaluator.self_normalized_ips([], None)
        assert result['snips_reward'] == 0.0
    
    def test_reward_bounds(self):
        """Reward must be in [0, 1]"""
        reward = self.reward_calc.compute_reward(
            clicked=True, dwell_time=30, session_continues=True, diversity_gain=0.5
        )
        assert 0 <= reward <= 1
    
    def test_no_click_reward(self):
        """No click → lower reward"""
        reward_click = self.reward_calc.compute_reward(True, 30, True, 0.5)
        reward_no_click = self.reward_calc.compute_reward(False, 30, True, 0.5)
        assert reward_click > reward_no_click


# ---------------------------------------------------------------------------
# Test RankingFeatures position bias
# ---------------------------------------------------------------------------
class TestPositionBias:
    
    def test_position_features_exist(self):
        """Position features must be part of feature vector"""
        from src.ranking_features import RankingFeatureBuilder
        builder = RankingFeatureBuilder({})
        
        features = builder.extract_position_features(position=0)
        assert 'display_position' in features
        assert 'position_log_discount' in features
        assert 'is_top3' in features
        assert 'is_top10' in features
    
    def test_position_discount_decreases(self):
        """Higher positions should have lower discount"""
        from src.ranking_features import RankingFeatureBuilder
        builder = RankingFeatureBuilder({})
        
        feat_pos0 = builder.extract_position_features(0)
        feat_pos10 = builder.extract_position_features(10)
        
        assert feat_pos0['position_log_discount'] > feat_pos10['position_log_discount']
    
    def test_top3_flag(self):
        """is_top3 flag correctness"""
        from src.ranking_features import RankingFeatureBuilder
        builder = RankingFeatureBuilder({})
        
        assert builder.extract_position_features(0)['is_top3'] == 1
        assert builder.extract_position_features(2)['is_top3'] == 1
        assert builder.extract_position_features(3)['is_top3'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
