"""
Base Ranking Models: CTR, Dwell, Retention
Produces calibrated signals for multi-objective optimization
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, mean_absolute_error
import lightgbm as lgb


class BaseRanker:
    """Base ranking model using LightGBM"""
    
    def __init__(self, config: Dict, task: str = 'binary'):
        """
        Args:
            config: Model configuration
            task: 'binary' for CTR/retention, 'regression' for dwell
        """
        self.config = config
        self.task = task
        self.model = None
        self.calibrated_model = None
        self.feature_names = []
        self.categorical_features = []
        self.label_encoders = {}
        
    def prepare_features(self, df: pd.DataFrame, 
                        feature_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for training
        
        Handles categorical encoding
        """
        df_features = df[feature_cols].copy()
        
        # Identify categorical columns
        categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
        
        # Label encode categorical features
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_features[col] = self.label_encoders[col].fit_transform(
                    df_features[col].astype(str)
                )
            else:
                df_features[col] = self.label_encoders[col].transform(
                    df_features[col].astype(str)
                )
        
        self.categorical_features = categorical_cols
        
        return df_features.values, categorical_cols
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             feature_names: List[str]):
        """
        Train LightGBM model
        """
        self.feature_names = feature_names
        
        # LightGBM parameters
        params = {
            'objective': 'binary' if self.task == 'binary' else 'regression',
            'metric': 'auc' if self.task == 'binary' else 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        # Create datasets
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            feature_name=feature_names,
            categorical_feature=self.categorical_features
        )
        
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            feature_name=feature_names,
            categorical_feature=self.categorical_features,
            reference=train_data
        )
        
        # Train
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
    
    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Calibrate model predictions using Platt scaling
        
        Only for binary classification tasks
        """
        if self.task != 'binary':
            return
        
        print("  Calibrating model...")
        
        # Wrap LightGBM for sklearn compatibility
        class LGBMWrapper:
            def __init__(self, model):
                self.model = model
            
            def predict_proba(self, X):
                preds = self.model.predict(X)
                return np.vstack([1 - preds, preds]).T
        
        wrapper = LGBMWrapper(self.model)
        
        self.calibrated_model = CalibratedClassifierCV(
            wrapper,
            method='sigmoid',
            cv='prefit'
        )
        
        self.calibrated_model.fit(X_val, y_val)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict scores
        
        Returns calibrated probabilities for binary, raw predictions for regression
        """
        if self.task == 'binary' and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance
        """
        preds = self.predict(X)
        
        metrics = {}
        
        if self.task == 'binary':
            metrics['auc'] = roc_auc_score(y, preds)
            metrics['logloss'] = log_loss(y, preds)
        else:
            metrics['rmse'] = np.sqrt(mean_squared_error(y, preds))
            metrics['mae'] = mean_absolute_error(y, preds)
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        importance = self.model.feature_importance(importance_type='gain')
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, save_path: str):
        """Save model"""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        self.model.save_model(str(path) + '_lgb.txt')
        
        # Save metadata
        metadata = {
            'task': self.task,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'label_encoders': self.label_encoders,
            'calibrated_model': self.calibrated_model
        }
        
        with open(str(path) + '_meta.pkl', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, save_path: str):
        """Load model"""
        path = Path(save_path)
        
        # Load LightGBM model
        self.model = lgb.Booster(model_file=str(path) + '_lgb.txt')
        
        # Load metadata
        with open(str(path) + '_meta.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.task = metadata['task']
        self.feature_names = metadata['feature_names']
        self.categorical_features = metadata['categorical_features']
        self.label_encoders = metadata['label_encoders']
        self.calibrated_model = metadata['calibrated_model']
