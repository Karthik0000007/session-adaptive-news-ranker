"""
Phase 2 Pipeline: Candidate Generation (Retrieval System)

Trains two-tower model and builds FAISS index for efficient retrieval
"""
import yaml
import pickle
import json
from pathlib import Path
from tqdm import tqdm

from src.data_loader import MINDDataLoader
from src.retrieval_system import RetrievalSystem
from src.retrieval_evaluator import RetrievalEvaluator


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add Phase 2 specific config
    config['two_tower'] = {
        'item_encoder': {
            'embedding_dim': 128,
            'text_weight': 0.7,
            'category_weight': 0.3
        },
        'user_encoder': {
            'embedding_dim': 128,
            'history_weight': 0.7,
            'session_weight': 0.3
        }
    }
    
    return config


def load_phase1_data(processed_dir: str):
    """Load Phase 1 outputs"""
    processed_path = Path(processed_dir)
    
    with open(processed_path / 'sessions.pkl', 'rb') as f:
        sessions = pickle.load(f)
    
    with open(processed_path / 'training_samples.pkl', 'rb') as f:
        training_samples = pickle.load(f)
    
    return sessions, training_samples


def prepare_evaluation_set(sessions: dict, 
                          article_metadata: dict,
                          sample_size: int = 1000) -> dict:
    """
    Prepare evaluation set from sessions
    
    Returns: {session_id: {clicked_items: [...]}}
    """
    eval_sessions = {}
    count = 0
    
    for user_id, user_sessions in sessions.items():
        for session_idx, session in enumerate(user_sessions):
            if count >= sample_size:
                break
            
            clicked_items = [
                event['article_id']
                for event in session
                if event['clicked'] == 1 and event['article_id'] in article_metadata
            ]
            
            if clicked_items:
                session_id = f"{user_id}_session_{session_idx}"
                eval_sessions[session_id] = {
                    'user_id': user_id,
                    'clicked_items': clicked_items,
                    'session_events': session
                }
                count += 1
        
        if count >= sample_size:
            break
    
    return eval_sessions


def main():
    print("=" * 60)
    print("Phase 2: Candidate Generation (Retrieval System)")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Setup paths
    raw_dir = Path(config['data']['raw_dir'])
    processed_dir = Path(config['data']['processed_dir'])
    retrieval_dir = processed_dir / 'retrieval_model'
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MIND data
    print("\n[1/5] Loading MIND dataset...")
    loader = MINDDataLoader(raw_dir)
    news_df = loader.load_news(config['data']['mind_news'])
    article_metadata = loader.get_article_metadata(news_df)
    print(f"  - Loaded {len(article_metadata)} articles")
    
    # Load Phase 1 data
    print("\n[2/5] Loading Phase 1 data...")
    sessions, training_samples = load_phase1_data(processed_dir)
    print(f"  - Loaded {len(sessions)} users")
    print(f"  - Loaded {len(training_samples)} training samples")
    
    # Initialize and train retrieval system
    print("\n[3/5] Training retrieval system...")
    retrieval_system = RetrievalSystem(config)
    retrieval_system.train(article_metadata)
    
    # Prepare evaluation set
    print("\n[4/5] Preparing evaluation set...")
    eval_sessions = prepare_evaluation_set(sessions, article_metadata, sample_size=500)
    print(f"  - Prepared {len(eval_sessions)} evaluation sessions")
    
    # Evaluate retrieval
    print("\n[5/5] Evaluating retrieval performance...")
    retrieval_results = {}
    
    for session_id, session_data in tqdm(eval_sessions.items(), desc="  Evaluating"):
        user_id = session_data['user_id']
        clicked_items = session_data['clicked_items']
        session_events = session_data['session_events']
        
        # Extract session state from last event
        if session_events:
            last_event = session_events[-1]
            session_state = {
                'session_length': len(session_events),
                'avg_dwell_time': sum(e.get('dwell_time', 0) for e in session_events) / len(session_events),
                'click_rate': sum(1 for e in session_events if e['clicked']) / len(session_events),
                'skip_rate': sum(1 for e in session_events if not e['clicked']) / len(session_events),
                'click_entropy': 0.0,  # Simplified
                'fatigue_score': 0.0   # Simplified
            }
        else:
            session_state = {}
        
        # Retrieve candidates
        candidates = retrieval_system.retrieve(
            user_id=user_id,
            clicked_article_ids=clicked_items,
            session_state=session_state,
            k=100
        )
        
        retrieval_results[session_id] = candidates
    
    # Compute metrics
    metrics = RetrievalEvaluator.evaluate_batch(
        eval_sessions,
        retrieval_results,
        k_values=[10, 50, 100]
    )
    
    RetrievalEvaluator.print_metrics(metrics)
    
    # Save retrieval system
    print("\nSaving retrieval system...")
    retrieval_system.save(str(retrieval_dir))
    
    # Save evaluation results
    eval_results = {
        'metrics': metrics,
        'num_eval_sessions': len(eval_sessions),
        'config': config['two_tower']
    }
    
    with open(retrieval_dir / 'eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Phase 2 Complete!")
    print("=" * 60)
    print(f"\nRetrieval system saved to: {retrieval_dir}")
    print("\nNext: Phase 3 - Base Ranking Models")


if __name__ == "__main__":
    main()
