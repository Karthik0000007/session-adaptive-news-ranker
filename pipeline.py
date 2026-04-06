"""
Main pipeline for Phase 1: Dataset & Session Simulation

Transforms MIND dataset into session-based training data
"""
import yaml
import pickle
import json
from pathlib import Path
from tqdm import tqdm

from src.data_loader import MINDDataLoader
from src.session_builder import SessionBuilder
from src.signal_simulator import SignalSimulator
from src.feature_extractor import FeatureExtractor
from src.negative_sampler import NegativeSampler


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    print("=" * 60)
    print("Phase 1: Dataset & Session Simulation")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Setup paths
    raw_dir = Path(config['data']['raw_dir'])
    processed_dir = Path(config['data']['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    print("\n[1/7] Initializing components...")
    loader = MINDDataLoader(raw_dir)
    session_builder = SessionBuilder(
        gap_minutes=config['session']['gap_minutes'],
        min_length=config['session']['min_session_length']
    )
    signal_simulator = SignalSimulator(config['simulation']['dwell_time'])
    feature_extractor = FeatureExtractor(config['features'])
    negative_sampler = NegativeSampler(config['simulation']['negative_samples'])
    
    # Load raw data
    print("\n[2/7] Loading MIND dataset...")
    behaviors_df = loader.load_behaviors(config['data']['mind_behaviors'])
    news_df = loader.load_news(config['data']['mind_news'])
    article_metadata = loader.get_article_metadata(news_df)
    
    print(f"  - Loaded {len(behaviors_df)} behavior records")
    print(f"  - Loaded {len(news_df)} news articles")
    
    # Parse impressions into events
    print("\n[3/7] Parsing impressions into events...")
    events = loader.parse_impressions(behaviors_df)
    print(f"  - Generated {len(events)} interaction events")
    
    # Build sessions
    print("\n[4/7] Building sessions...")
    sessions = session_builder.build_sessions(events)
    session_stats = session_builder.get_session_stats(sessions)
    
    print(f"  - Total users: {session_stats['total_users']}")
    print(f"  - Total sessions: {session_stats['total_sessions']}")
    print(f"  - Avg sessions per user: {session_stats['avg_sessions_per_user']:.2f}")
    print(f"  - Avg session length: {session_stats['avg_session_length']:.2f}")
    
    # Simulate signals
    print("\n[5/7] Simulating behavioral signals...")
    enriched_sessions = {}
    
    for user_id, user_sessions in tqdm(sessions.items(), desc="  Processing users"):
        enriched_user_sessions = []
        
        for session in user_sessions:
            enriched_session = signal_simulator.add_signals_to_session(session)
            enriched_user_sessions.append(enriched_session)
        
        enriched_sessions[user_id] = enriched_user_sessions
    
    # Extract features and create training samples
    print("\n[6/7] Extracting features and creating training samples...")
    training_samples = []
    all_article_ids = list(article_metadata.keys())
    
    for user_id, user_sessions in tqdm(enriched_sessions.items(), desc="  Processing sessions"):
        for session in user_sessions:
            seen_articles = set()
            
            for idx, event in enumerate(session):
                # Extract session state at this point
                session_state = feature_extractor.extract_session_state(
                    session, idx, article_metadata
                )
                
                # Add fatigue score
                session_state['fatigue_score'] = signal_simulator.compute_fatigue_score(
                    session, idx
                )
                
                # Create positive sample
                positive_sample = feature_extractor.create_training_sample(
                    user_id=user_id,
                    article_id=event['article_id'],
                    session_state=session_state,
                    article_metadata=article_metadata,
                    label_click=event['clicked'],
                    label_dwell=event['dwell_time']
                )
                training_samples.append(positive_sample)
                
                # Generate negatives for clicked items
                if event['clicked'] == 1:
                    negatives = negative_sampler.sample_negatives(
                        positive_article=event['article_id'],
                        candidate_pool=all_article_ids,
                        exclude_set=seen_articles
                    )
                    
                    for neg_article in negatives:
                        neg_sample = feature_extractor.create_training_sample(
                            user_id=user_id,
                            article_id=neg_article,
                            session_state=session_state,
                            article_metadata=article_metadata,
                            label_click=0,
                            label_dwell=0.0
                        )
                        training_samples.append(neg_sample)
                
                seen_articles.add(event['article_id'])
    
    print(f"  - Generated {len(training_samples)} training samples")
    
    # Save outputs
    print("\n[7/7] Saving outputs...")
    
    # Save sessions
    sessions_path = processed_dir / config['output']['sessions_file']
    with open(sessions_path, 'wb') as f:
        pickle.dump(enriched_sessions, f)
    print(f"  - Saved sessions to {sessions_path}")
    
    # Save training samples
    features_path = processed_dir / config['output']['features_file']
    with open(features_path, 'wb') as f:
        pickle.dump(training_samples, f)
    print(f"  - Saved training samples to {features_path}")
    
    # Save stats
    stats = {
        'session_stats': session_stats,
        'total_training_samples': len(training_samples),
        'total_articles': len(article_metadata),
        'config': config
    }
    
    stats_path = processed_dir / config['output']['stats_file']
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  - Saved statistics to {stats_path}")
    
    print("\n" + "=" * 60)
    print("Phase 1 Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {processed_dir}")
    print("\nNext: Phase 2 - Feature Engineering & Candidate Generation")


if __name__ == "__main__":
    main()
