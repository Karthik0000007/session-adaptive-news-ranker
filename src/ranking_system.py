"""
Ranking System: Orchestrates multiple ranking models
Provides unified prediction interface
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path

from src.base_ranker import BaseRanker
from src.ranking_features import RankingFeatureBuilder


class RankingSystem:
    """Multi-objective ranking system"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize models
        self.ctr_model = BaseRanker(config, task='binary')
        self.dwell_model = BaseRanker(config, task='regression')
        self.retention_model = BaseRanker(config, task='binary')
        
        # Feature builder
        self.feature_builder = None
        self.feature_cols = []
        
    def train(self, 
             training_df: pd.DataFrame,
             feature_builder: RankingFeatureBuilder):
        """
        Train all ranking models
        
        Args:
            training_df: DataFrame with features and labels
            feature_builder: Fitted feature builder
        """
        self.feature_builder = feature_builder
        
        # Identify feature columns
        exclude_cols = ['label_click', 'label_dwell', 'user_id', 'article_id']
        self.feature_cols = [c for c in training_df.columns if c not in exclude_cols]
        
        print(f"Training with {len(self.feature_cols)} features")
        
        # Time-based split (simulate production)
        # Use first 80% for train, last 20% for validation
        split_idx = int(len(training_df) * 0.8)
        train_df = training_df.iloc[:split_idx]
        val_df = training_df.iloc[split_idx:]
        
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
        
        # Train CTR model
        print("\n" + "=" * 60)
        print("Training CTR Model")
        print("=" * 60)
        self._train_ctr_model(train_df, val_df)
        
        # Train Dwell model
        print("\n" + "=" * 60)
        print("Training Dwell Model")
        print("=" * 60)
        self._train_dwell_model(train_df, val_df)
        
        # Train Retention model (simplified: session continuation)
        print("\n" + "=" * 60)
        print("Training Retention Model")
        print("=" * 60)
        self._train_retention_model(train_df, val_df)
    
    def _train_ctr_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train CTR prediction model"""
        X_train, cat_features = self.ctr_model.prepare_features(train_df, self.feature_cols)
        y_train = train_df['label_click'].values
        
        X_val, _ = self.ctr_model.prepare_features(val_df, self.feature_cols)
        y_val = val_df['label_click'].values
        
        self.ctr_model.train(X_train, y_train, X_val, y_val, self.feature_cols)
        self.ctr_model.calibrate(X_val, y_val)
        
        # Evaluate
        metrics = self.ctr_model.evaluate(X_val, y_val)
        print(f"\nCTR Model Validation Metrics:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  LogLoss: {metrics['logloss']:.4f}")
    
    def _train_dwell_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train dwell time prediction model"""
        # Only train on clicked items
        train_clicked = train_df[train_df['label_click'] == 1].copy()
        val_clicked = val_df[val_df['label_click'] == 1].copy()
        
        if len(train_clicked) == 0 or len(val_clicked) == 0:
            print("  Warning: No clicked items for dwell model")
            return
        
        X_train, _ = self.dwell_model.prepare_features(train_clicked, self.feature_cols)
        y_train = np.log1p(train_clicked['label_dwell'].values)  # Log transform
        
        X_val, _ = self.dwell_model.prepare_features(val_clicked, self.feature_cols)
        y_val = np.log1p(val_clicked['label_dwell'].values)
        
        self.dwell_model.train(X_train, y_train, X_val, y_val, self.feature_cols)
        
        # Evaluate
        metrics = self.dwell_model.evaluate(X_val, y_val)
        print(f"\nDwell Model Validation Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
    
    def _train_retention_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Train retention prediction model
        
        Simplified: Predict if user will click again (proxy for session continuation)
        """
        # Create retention label (simplified)
        # In production, this would be actual session continuation
        train_df['label_retention'] = (train_df['label_click'] > 0).astype(int)
        val_df['label_retention'] = (val_df['label_click'] > 0).astype(int)
        
        X_train, _ = self.retention_model.prepare_features(train_df, self.feature_cols)
        y_train = train_df['label_retention'].values
        
        X_val, _ = self.retention_model.prepare_features(val_df, self.feature_cols)
        y_val = val_df['label_retention'].values
        
        self.retention_model.train(X_train, y_train, X_val, y_val, self.feature_cols)
        self.retention_model.calibrate(X_val, y_val)
        
        # Evaluate
        metrics = self.retention_model.evaluate(X_val, y_val)
        print(f"\nRetention Model Validation Metrics:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  LogLoss: {metrics['logloss']:.4f}")
    
    def predict(self,
               user_id: str,
               article_id: str,
               session_state: Dict,
               article_metadata: Dict[str, Dict],
               clicked_history: List[str]) -> Dict[str, float]:
        """
        Predict all objectives for a candidate
        
        Returns: {
            'ctr_score': float,
            'dwell_score': float,
            'retention_score': float
        }
        """
        # Build features
        features = self.feature_builder.build_features(
            user_id, article_id, session_state, article_metadata, clicked_history
        )
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Prepare features
        X_ctr, _ = self.ctr_model.prepare_features(df, self.feature_cols)
        X_dwell, _ = self.dwell_model.prepare_features(df, self.feature_cols)
        X_retention, _ = self.retention_model.prepare_features(df, self.feature_cols)
        
        # Predict
        ctr_score = self.ctr_model.predict(X_ctr)[0]
        dwell_score = np.expm1(self.dwell_model.predict(X_dwell)[0])  # Inverse log transform
        retention_score = self.retention_model.predict(X_retention)[0]
        
        # Normalize dwell score to [0, 1]
        dwell_score_norm = min(dwell_score / 300.0, 1.0)  # Cap at 5 minutes
        
        return {
            'ctr_score': float(ctr_score),
            'dwell_score': float(dwell_score_norm),
            'retention_score': float(retention_score)
        }
    
    def predict_batch(self,
                     candidates: List[Dict]) -> List[Dict[str, float]]:
        """
        Predict for multiple candidates
        
        Args:
            candidates: List of {user_id, article_id, session_state, ...}
        
        Returns: List of prediction dicts
        """
        predictions = []
        
        for candidate in candidates:
            pred = self.predict(
                user_id=candidate['user_id'],
                article_id=candidate['article_id'],
                session_state=candidate['session_state'],
                article_metadata=candidate['article_metadata'],
                clicked_history=candidate.get('clicked_history', [])
            )
            predictions.append(pred)
        
        return predictions
    
    def save(self, save_dir: str):
        """Save ranking system"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.ctr_model.save(str(save_path / 'ctr_model'))
        self.dwell_model.save(str(save_path / 'dwell_model'))
        self.retention_model.save(str(save_path / 'retention_model'))
        
        print(f"Ranking system saved to {save_path}")
    
    def load(self, save_dir: str):
        """Load ranking system"""
        save_path = Path(save_dir)
        
        self.ctr_model.load(str(save_path / 'ctr_model'))
        self.dwell_model.load(str(save_path / 'dwell_model'))
        self.retention_model.load(str(save_path / 'retention_model'))
        
        print(f"Ranking system loaded from {save_path}")
