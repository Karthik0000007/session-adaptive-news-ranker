"""
Phase 8: A/B Testing Simulation Framework

Simulates user behavior and compares ranking strategies:
- Baseline (fixed weights)
- Rule-based (session-adaptive)
- Bandit (LinUCB)
- Engagement-only (control)
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import logging

from src.retrieval_system import RetrievalSystem
from src.ranking_system import RankingSystem
from src.decision_layer import DecisionLayer
from src.contextual_bandit import ContextualBandit
from src.weight_adapter import WeightAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for A/B test simulation"""
    num_sessions: int = 1000
    max_session_length: int = 20
    random_seed: int = 42
    strategies: List[str] = None
    
    def __post_init__(self):
        if self.strategies is None:
            self.strategies = ['baseline', 'rule_based', 'bandit', 'engagement_only']


class UserSimulator:
    """Simulates realistic user behavior"""
    
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)
        
    def simulate_click(self, ctr_score: float, position: int) -> bool:
        """Simulate click with position bias"""
        position_discount = 1.0 / np.log2(position + 2)
        click_prob = self._sigmoid(ctr_score) * position_discount
        noise = self.rng.normal(0, 0.1)
        return self.rng.random() < np.clip(click_prob + noise, 0, 1)
    
    def simulate_dwell(self, dwell_score: float, clicked: bool) -> float:
        """Simulate dwell time"""
        if not clicked:
            return self.rng.uniform(0.5, 2.0)
        
        mean_dwell = dwell_score * 100  # Scale to seconds
        return self.rng.lognormal(np.log(mean_dwell + 1), 0.5)
    
    def simulate_continuation(self, avg_dwell: float, diversity: float, 
                            fatigue: float) -> bool:
        """Simulate whether user continues session"""
        continue_prob = 0.3 + 0.3 * (avg_dwell / 60) + 0.2 * diversity - 0.4 * fatigue
        return self.rng.random() < np.clip(continue_prob, 0, 1)
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))


class ABTestSimulator:
    """Main A/B testing simulation engine"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.user_sim = UserSimulator(config.random_seed)
        
        # Load systems
        self.retrieval = self._load_retrieval()
        self.ranking = self._load_ranking()
        
        # Initialize strategies
        self.strategies = self._init_strategies()
        
        # Results storage
        self.results = {strategy: [] for strategy in config.strategies}
        
    def _load_retrieval(self) -> RetrievalSystem:
        """Load retrieval system"""
        model_dir = Path('data/processed/retrieval_model')
        return RetrievalSystem.load(model_dir)
    
    def _load_ranking(self) -> RankingSystem:
        """Load ranking system"""
        model_dir = Path('data/processed/ranking_models')
        return RankingSystem.load(model_dir)
    
    def _init_strategies(self) -> Dict:
        """Initialize all ranking strategies"""
        strategies = {}
        
        # Baseline: fixed weights
        if 'baseline' in self.config.strategies:
            strategies['baseline'] = DecisionLayer(
                ranking_system=self.ranking,
                weight_adapter=None,
                fixed_weights=[0.4, 0.3, 0.2, 0.1]
            )
        
        # Rule-based: session-adaptive
        if 'rule_based' in self.config.strategies:
            strategies['rule_based'] = DecisionLayer(
                ranking_system=self.ranking,
                weight_adapter=WeightAdapter(),
                fixed_weights=None
            )
        
        # Bandit: LinUCB
        if 'bandit' in self.config.strategies:
            bandit_path = Path('data/processed/bandit_model/bandit_state.pkl')
            if bandit_path.exists():
                with open(bandit_path, 'rb') as f:
                    bandit = pickle.load(f)
            else:
                bandit = ContextualBandit(alpha=0.5)
            
            strategies['bandit'] = {
                'decision_layer': DecisionLayer(
                    ranking_system=self.ranking,
                    weight_adapter=None,
                    fixed_weights=None
                ),
                'bandit': bandit
            }
        
        # Engagement-only: no diversity/novelty
        if 'engagement_only' in self.config.strategies:
            strategies['engagement_only'] = DecisionLayer(
                ranking_system=self.ranking,
                weight_adapter=None,
                fixed_weights=[0.6, 0.2, 0.1, 0.1]
            )
        
        return strategies
    
    def simulate_session(self, strategy_name: str, user_data: Dict) -> Dict:
        """Simulate a single user session"""
        session_metrics = {
            'clicks': 0,
            'total_dwell': 0.0,
            'interactions': 0,
            'diversity_scores': [],
            'novelty_scores': [],
            'weights_history': []
        }
        
        session_state = self._init_session_state()
        
        for t in range(self.config.max_session_length):
            # Get candidates
            candidates = self.retrieval.retrieve(
                user_data['user_id'],
                session_state,
                k=100
            )
            
            if not candidates:
                break
            
            # Rank with strategy
            ranked_list, weights = self._rank_with_strategy(
                strategy_name, candidates, session_state
            )
            
            session_metrics['weights_history'].append(weights)
            
            # Simulate user interaction
            interaction = self._simulate_interaction(
                ranked_list[:20], session_state
            )
            
            # Update metrics
            session_metrics['clicks'] += interaction['clicked']
            session_metrics['total_dwell'] += interaction['dwell_time']
            session_metrics['interactions'] += 1
            session_metrics['diversity_scores'].append(interaction['diversity'])
            session_metrics['novelty_scores'].append(interaction['novelty'])
            
            # Update session state
            session_state = self._update_session_state(
                session_state, interaction
            )
            
            # Check continuation
            if not self.user_sim.simulate_continuation(
                session_state['avg_dwell_time'],
                interaction['diversity'],
                session_state['fatigue']
            ):
                break
        
        return self._compute_session_summary(session_metrics)
    
    def _rank_with_strategy(self, strategy_name: str, candidates: List,
                           session_state: Dict) -> Tuple[List, List]:
        """Rank candidates using specified strategy"""
        strategy = self.strategies[strategy_name]
        
        if strategy_name == 'bandit':
            # Use bandit to select weights
            bandit = strategy['bandit']
            action_idx = bandit.select_action(session_state)
            weights = bandit.actions[action_idx]
            
            decision_layer = strategy['decision_layer']
            decision_layer.fixed_weights = weights
            ranked_list = decision_layer.rank(candidates, session_state)
        else:
            ranked_list = strategy.rank(candidates, session_state)
            weights = strategy.get_current_weights()
        
        return ranked_list, weights
    
    def _simulate_interaction(self, ranked_list: List, 
                             session_state: Dict) -> Dict:
        """Simulate user interaction with ranked list"""
        clicked = False
        dwell_time = 0.0
        clicked_item = None
        
        for position, item in enumerate(ranked_list):
            if self.user_sim.simulate_click(item['ctr_score'], position):
                clicked = True
                clicked_item = item
                dwell_time = self.user_sim.simulate_dwell(
                    item['dwell_score'], True
                )
                break
        
        if not clicked and ranked_list:
            # User scanned but didn't click
            dwell_time = self.user_sim.simulate_dwell(0.0, False)
        
        # Compute diversity and novelty
        diversity = self._compute_diversity(ranked_list)
        novelty = np.mean([item.get('novelty_score', 0.5) 
                          for item in ranked_list[:5]])
        
        return {
            'clicked': 1 if clicked else 0,
            'dwell_time': dwell_time,
            'clicked_item': clicked_item,
            'diversity': diversity,
            'novelty': novelty
        }
    
    @staticmethod
    def _compute_diversity(ranked_list: List) -> float:
        """Compute diversity of ranked list"""
        if len(ranked_list) < 2:
            return 0.0
        
        categories = [item.get('category', 'unknown') for item in ranked_list[:10]]
        unique_ratio = len(set(categories)) / len(categories)
        return unique_ratio
    
    @staticmethod
    def _init_session_state() -> Dict:
        """Initialize session state"""
        return {
            'session_length': 0,
            'avg_dwell_time': 0.0,
            'click_rate': 0.0,
            'skip_rate': 0.0,
            'click_entropy': 0.0,
            'fatigue': 0.0,
            'time_of_day': 12,
            'day_of_week': 3
        }
    
    @staticmethod
    def _update_session_state(state: Dict, interaction: Dict) -> Dict:
        """Update session state after interaction"""
        state['session_length'] += 1
        
        # Update dwell time
        total_dwell = state['avg_dwell_time'] * (state['session_length'] - 1)
        state['avg_dwell_time'] = (total_dwell + interaction['dwell_time']) / state['session_length']
        
        # Update click rate
        total_clicks = state['click_rate'] * (state['session_length'] - 1)
        state['click_rate'] = (total_clicks + interaction['clicked']) / state['session_length']
        
        # Update skip rate
        state['skip_rate'] = 1 - state['click_rate']
        
        # Update fatigue (increases with session length)
        state['fatigue'] = min(1.0, state['session_length'] / 20)
        
        return state
    
    @staticmethod
    def _compute_session_summary(metrics: Dict) -> Dict:
        """Compute summary statistics for session"""
        return {
            'ctr': metrics['clicks'] / max(metrics['interactions'], 1),
            'avg_dwell': metrics['total_dwell'] / max(metrics['interactions'], 1),
            'session_length': metrics['interactions'],
            'avg_diversity': np.mean(metrics['diversity_scores']) if metrics['diversity_scores'] else 0.0,
            'avg_novelty': np.mean(metrics['novelty_scores']) if metrics['novelty_scores'] else 0.0,
            'weights_history': metrics['weights_history']
        }
    
    def run_experiment(self) -> Dict:
        """Run full A/B test simulation"""
        logger.info(f"Starting A/B test with {self.config.num_sessions} sessions")
        
        # Load test users
        test_users = self._load_test_users()
        
        for strategy in self.config.strategies:
            logger.info(f"Simulating strategy: {strategy}")
            
            for i, user_data in enumerate(test_users[:self.config.num_sessions]):
                if (i + 1) % 100 == 0:
                    logger.info(f"  Completed {i + 1} sessions")
                
                session_result = self.simulate_session(strategy, user_data)
                self.results[strategy].append(session_result)
        
        # Compute aggregate results
        return self._aggregate_results()
    
    def _load_test_users(self) -> List[Dict]:
        """Load test users from processed data"""
        sessions_path = Path('data/processed/sessions.pkl')
        with open(sessions_path, 'rb') as f:
            sessions = pickle.load(f)
        
        # Extract unique users
        users = []
        seen_users = set()
        for session in sessions:
            user_id = session['user_id']
            if user_id not in seen_users:
                users.append({'user_id': user_id, 'history': session.get('history', [])})
                seen_users.add(user_id)
        
        return users
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results across all sessions"""
        aggregated = {}
        
        for strategy, sessions in self.results.items():
            aggregated[strategy] = {
                'ctr': np.mean([s['ctr'] for s in sessions]),
                'ctr_std': np.std([s['ctr'] for s in sessions]),
                'avg_dwell': np.mean([s['avg_dwell'] for s in sessions]),
                'avg_dwell_std': np.std([s['avg_dwell'] for s in sessions]),
                'session_length': np.mean([s['session_length'] for s in sessions]),
                'session_length_std': np.std([s['session_length'] for s in sessions]),
                'diversity': np.mean([s['avg_diversity'] for s in sessions]),
                'diversity_std': np.std([s['avg_diversity'] for s in sessions]),
                'novelty': np.mean([s['avg_novelty'] for s in sessions]),
                'novelty_std': np.std([s['avg_novelty'] for s in sessions])
            }
        
        return aggregated


def print_comparison_table(results: Dict):
    """Print formatted comparison table"""
    print("\n" + "="*80)
    print("A/B TEST RESULTS")
    print("="*80)
    
    print(f"\n{'Strategy':<20} {'CTR':<12} {'Dwell (s)':<12} {'Session Len':<12} {'Diversity':<12}")
    print("-"*80)
    
    for strategy, metrics in results.items():
        print(f"{strategy:<20} "
              f"{metrics['ctr']:.4f}±{metrics['ctr_std']:.4f}  "
              f"{metrics['avg_dwell']:.2f}±{metrics['avg_dwell_std']:.2f}  "
              f"{metrics['session_length']:.2f}±{metrics['session_length_std']:.2f}  "
              f"{metrics['diversity']:.4f}±{metrics['diversity_std']:.4f}")
    
    print("="*80)


def main():
    """Main execution"""
    config = SimulationConfig(
        num_sessions=1000,
        max_session_length=20,
        random_seed=42
    )
    
    simulator = ABTestSimulator(config)
    results = simulator.run_experiment()
    
    # Print results
    print_comparison_table(results)
    
    # Save results
    output_dir = Path('data/processed/ab_test_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
