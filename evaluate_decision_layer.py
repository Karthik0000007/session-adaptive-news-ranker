"""
Phase 4 & 5 Pipeline: Multi-Objective Scoring + Rule-Based Weighting

Tests both fixed and adaptive weight strategies
"""
import yaml
import pickle
import json
from pathlib import Path
import numpy as np

from src.data_loader import MINDDataLoader
from src.retrieval_system import RetrievalSystem
from src.ranking_system import RankingSystem
from src.decision_layer import DecisionLayer, RankingPipeline, WeightStrategy


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add Phase 4 & 5 config
    config['scoring'] = {
        'novelty_beta': 0.7,
        'freshness_decay': 0.1
    }
    
    config['weighting'] = {
        'early_session_threshold': 3,
        'high_engagement_threshold': 30.0,
        'fatigue_threshold': 0.6,
        'smoothing_alpha': 0.7
    }
    
    config['weight_strategy'] = WeightStrategy.RULE_BASED
    
    return config


def load_phase1_data(processed_dir: str):
    """Load Phase 1 outputs"""
    processed_path = Path(processed_dir)
    
    with open(processed_path / 'sessions.pkl', 'rb') as f:
        sessions = pickle.load(f)
    
    return sessions


def evaluate_strategy(pipeline: RankingPipeline,
                     test_sessions: dict,
                     article_metadata: dict,
                     strategy: str) -> dict:
    """
    Evaluate ranking pipeline with given strategy
    """
    pipeline.decision_layer.set_strategy(strategy)
    
    metrics = {
        'avg_engagement': [],
        'avg_retention': [],
        'avg_diversity': [],
        'avg_novelty': []
    }
    
    num_sessions = min(50, len(test_sessions))
    
    for idx, (user_id, user_sessions) in enumerate(test_sessions.items()):
        if idx >= num_sessions:
            break
        
        for session in user_sessions[:1]:  # First session per user
            # Build session state
            clicked_ids = [e['article_id'] for e in session if e['clicked'] == 1]
            
            if not clicked_ids:
                continue
            
            session_state = {
                'session_length': len(session),
                'avg_dwell_time': np.mean([e.get('dwell_time', 0) for e in session]),
                'click_rate': sum(1 for e in session if e['clicked']) / len(session),
                'skip_rate': sum(1 for e in session if not e['clicked']) / len(session),
                'click_entropy': 0.5,
                'fatigue_score': 0.3,
                'time_of_day': 14,
                'day_of_week': 2
            }
            
            try:
                # Rank
                ranked = pipeline.rank(
                    user_id=user_id,
                    clicked_article_ids=clicked_ids,
                    session_state=session_state,
                    article_metadata=article_metadata,
                    k=20
                )
                
                # Collect component scores
                for _, _, components in ranked:
                    metrics['avg_engagement'].append(components['engagement'])
                    metrics['avg_retention'].append(components['retention'])
                    metrics['avg_diversity'].append(components['diversity'])
                    metrics['avg_novelty'].append(components['novelty'])
            
            except Exception as e:
                continue
    
    # Average metrics
    results = {}
    for key, values in metrics.items():
        if values:
            results[key] = np.mean(values)
        else:
            results[key] = 0.0
    
    return results


def main():
    print("=" * 60)
    print("Phase 4 & 5: Multi-Objective Scoring + Adaptive Weighting")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Setup paths
    raw_dir = Path(config['data']['raw_dir'])
    processed_dir = Path(config['data']['processed_dir'])
    
    # Load MIND data
    print("\n[1/5] Loading MIND dataset...")
    loader = MINDDataLoader(raw_dir)
    news_df = loader.load_news(config['data']['mind_news'])
    article_metadata = loader.get_article_metadata(news_df)
    print(f"  - Loaded {len(article_metadata)} articles")
    
    # Load Phase 1 data
    print("\n[2/5] Loading Phase 1 data...")
    sessions = load_phase1_data(processed_dir)
    print(f"  - Loaded {len(sessions)} users")
    
    # Load retrieval system
    print("\n[3/5] Loading retrieval system...")
    retrieval_system = RetrievalSystem(config)
    retrieval_system.load(str(processed_dir / 'retrieval_model'))
    
    # Load ranking system
    print("\n[4/5] Loading ranking system...")
    ranking_system = RankingSystem(config.get('ranking', {}))
    ranking_system.load(str(processed_dir / 'ranking_models'))
    
    with open(processed_dir / 'ranking_models' / 'feature_builder.pkl', 'rb') as f:
        ranking_system.feature_builder = pickle.load(f)
    
    # Initialize decision layer
    print("\n[5/5] Initializing decision layer...")
    decision_layer = DecisionLayer(config)
    
    # Create full pipeline
    pipeline = RankingPipeline(
        retrieval_system=retrieval_system,
        ranking_system=ranking_system,
        decision_layer=decision_layer
    )
    
    print("\n" + "=" * 60)
    print("Testing Weight Strategies")
    print("=" * 60)
    
    # Test sessions
    test_sessions = dict(list(sessions.items())[:20])
    
    # Evaluate fixed weights
    print("\n[Strategy 1] Fixed Baseline Weights")
    print("-" * 60)
    fixed_metrics = evaluate_strategy(
        pipeline, test_sessions, article_metadata, WeightStrategy.FIXED
    )
    
    print("Results:")
    for metric, value in fixed_metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    # Evaluate rule-based weights
    print("\n[Strategy 2] Rule-Based Adaptive Weights")
    print("-" * 60)
    adaptive_metrics = evaluate_strategy(
        pipeline, test_sessions, article_metadata, WeightStrategy.RULE_BASED
    )
    
    print("Results:")
    for metric, value in adaptive_metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    # Compare
    print("\n" + "=" * 60)
    print("Strategy Comparison")
    print("=" * 60)
    
    for metric in fixed_metrics.keys():
        fixed_val = fixed_metrics[metric]
        adaptive_val = adaptive_metrics[metric]
        diff = adaptive_val - fixed_val
        pct_change = (diff / fixed_val * 100) if fixed_val > 0 else 0
        
        print(f"{metric:20s}: Fixed={fixed_val:.4f}, Adaptive={adaptive_val:.4f}, "
              f"Δ={diff:+.4f} ({pct_change:+.1f}%)")
    
    # Save results
    results = {
        'fixed_weights': fixed_metrics,
        'adaptive_weights': adaptive_metrics,
        'config': {
            'scoring': config['scoring'],
            'weighting': config['weighting']
        }
    }
    
    output_dir = processed_dir / 'decision_layer'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    print("\n" + "=" * 60)
    print("Phase 4 & 5 Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Multi-objective scoring combines engagement, retention, diversity, novelty")
    print("- Rule-based adaptation adjusts weights based on session state")
    print("- Greedy reranking ensures diversity in final list")
    print("\nNext: Phase 6 - Contextual Bandit (Learned Weights)")


if __name__ == "__main__":
    main()
