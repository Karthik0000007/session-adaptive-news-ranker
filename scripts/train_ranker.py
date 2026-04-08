"""
Phase 3 Pipeline: Base Ranking Models

Trains CTR, Dwell, and Retention models
"""
import yaml
import pickle
import json
from pathlib import Path

from src.data_loader import MINDDataLoader
from src.ranking_features import RankingFeatureBuilder
from src.ranking_system import RankingSystem


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add Phase 3 specific config
    config['ranking'] = {
        'model_type': 'lightgbm',
        'calibration': True
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


def main():
    print("=" * 60)
    print("Phase 3: Base Ranking Models")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Setup paths
    raw_dir = Path(config['data']['raw_dir'])
    processed_dir = Path(config['data']['processed_dir'])
    ranking_dir = processed_dir / 'ranking_models'
    ranking_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Build feature engineering pipeline
    print("\n[3/5] Building feature engineering pipeline...")
    feature_builder = RankingFeatureBuilder(config)
    feature_builder.build_user_profiles(sessions)
    feature_builder.build_item_stats(article_metadata, sessions)
    
    # Prepare training data
    print("\n[4/5] Preparing training data...")
    training_df = feature_builder.prepare_training_data(
        training_samples,
        article_metadata
    )
    
    print(f"\nTraining Data Summary:")
    print(f"  - Total samples: {len(training_df)}")
    print(f"  - Positive clicks: {training_df['label_click'].sum()}")
    print(f"  - Click rate: {training_df['label_click'].mean():.4f}")
    print(f"  - Avg dwell (clicked): {training_df[training_df['label_click']==1]['label_dwell'].mean():.2f}s")
    
    # Train ranking system
    print("\n[5/5] Training ranking models...")
    ranking_system = RankingSystem(config['ranking'])
    ranking_system.train(training_df, feature_builder)
    
    # Save models
    print("\nSaving ranking system...")
    ranking_system.save(str(ranking_dir))
    
    # Save feature builder
    with open(ranking_dir / 'feature_builder.pkl', 'wb') as f:
        pickle.dump(feature_builder, f)
    
    # Get feature importance
    print("\n" + "=" * 60)
    print("Top 10 Features by Importance")
    print("=" * 60)
    
    print("\nCTR Model:")
    ctr_importance = ranking_system.ctr_model.get_feature_importance()
    print(ctr_importance.head(10).to_string(index=False))
    
    print("\nDwell Model:")
    dwell_importance = ranking_system.dwell_model.get_feature_importance()
    print(dwell_importance.head(10).to_string(index=False))
    
    print("\nRetention Model:")
    retention_importance = ranking_system.retention_model.get_feature_importance()
    print(retention_importance.head(10).to_string(index=False))
    
    # Save summary
    summary = {
        'num_training_samples': len(training_df),
        'num_features': len(ranking_system.feature_cols),
        'click_rate': float(training_df['label_click'].mean()),
        'avg_dwell_clicked': float(training_df[training_df['label_click']==1]['label_dwell'].mean()),
        'config': config['ranking']
    }
    
    with open(ranking_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Phase 3 Complete!")
    print("=" * 60)
    print(f"\nRanking models saved to: {ranking_dir}")
    print("\nNext: Phase 4 - Multi-Objective Scoring")


if __name__ == "__main__":
    main()
