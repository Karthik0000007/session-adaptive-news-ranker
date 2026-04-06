"""
Contextual Bandit: LinUCB for dynamic weight selection
"""
import numpy as np
from typing import Dict, List, Tuple


class LinUCB:
    """
    Linear Upper Confidence Bound (LinUCB) algorithm
    
    Learns optimal weight selection policy from interaction data
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Discrete action space (weight vectors)
        self.actions = [
            [0.4, 0.3, 0.2, 0.1],  # Balanced
            [0.6, 0.2, 0.1, 0.1],  # Engagement-heavy
            [0.2, 0.3, 0.3, 0.2],  # Diversity/retention
            [0.2, 0.2, 0.2, 0.4],  # Novelty-heavy
        ]
        
        # Exploration parameter
        self.alpha = config.get('alpha', 0.5)
        
        # Feature dimension (will be set on first update)
        self.d = None
        
        # Per-action parameters
        self.A = {}  # A_a: d x d matrix
        self.b = {}  # b_a: d x 1 vector
        
        # Initialize for each action
        for idx in range(len(self.actions)):
            self.A[idx] = None
            self.b[idx] = None
    
    def _initialize_action(self, action_idx: int, d: int):
        """Initialize parameters for an action"""
        self.A[action_idx] = np.identity(d)
        self.b[action_idx] = np.zeros(d)
    
    def featurize(self, state: Dict) -> np.ndarray:
        """
        Convert session state to feature vector
        
        Features:
        - session_length (normalized)
        - avg_dwell_time (normalized)
        - click_rate
        - skip_rate
        - click_entropy (normalized)
        - fatigue_score
        - time_of_day (normalized)
        - day_of_week (normalized)
        """
        features = [
            min(state.get('session_length', 0) / 50.0, 1.0),
            min(state.get('avg_dwell_time', 0) / 300.0, 1.0),
            state.get('click_rate', 0.0),
            state.get('skip_rate', 0.0),
            min(state.get('click_entropy', 0) / 3.0, 1.0),
            state.get('fatigue_score', 0.0),
            state.get('time_of_day', 12) / 24.0,
            state.get('day_of_week', 3) / 7.0
        ]
        
        # Add bias term
        features.append(1.0)
        
        return np.array(features, dtype=np.float64)
    
    def select_action(self, state: Dict, explore: bool = True) -> Tuple[int, List[float]]:
        """
        Select action (weight vector) for given state
        
        Args:
            state: Session state dict
            explore: If True, use UCB; if False, use greedy
        
        Returns: (action_idx, weights)
        """
        x = self.featurize(state)
        
        # Initialize if first call
        if self.d is None:
            self.d = len(x)
            for idx in range(len(self.actions)):
                self._initialize_action(idx, self.d)
        
        # Compute score for each action
        scores = []
        
        for action_idx in range(len(self.actions)):
            # Compute theta_a = A_a^{-1} b_a
            A_inv = np.linalg.inv(self.A[action_idx])
            theta = A_inv @ self.b[action_idx]
            
            # Exploitation term
            exploitation = theta.T @ x
            
            # Exploration term (UCB)
            if explore:
                uncertainty = np.sqrt(x.T @ A_inv @ x)
                exploration = self.alpha * uncertainty
            else:
                exploration = 0.0
            
            score = exploitation + exploration
            scores.append(score)
        
        # Select best action
        best_action_idx = int(np.argmax(scores))
        best_weights = self.actions[best_action_idx]
        
        return best_action_idx, best_weights
    
    def update(self, state: Dict, action_idx: int, reward: float):
        """
        Update bandit parameters based on observed reward
        
        Args:
            state: Session state
            action_idx: Selected action index
            reward: Observed reward
        """
        x = self.featurize(state)
        
        # Update A_a and b_a
        self.A[action_idx] += np.outer(x, x)
        self.b[action_idx] += reward * x
    
    def get_action_weights(self, action_idx: int) -> List[float]:
        """Get weight vector for action index"""
        return self.actions[action_idx]
    
    def get_action_probabilities(self, state: Dict) -> np.ndarray:
        """
        Get probability distribution over actions (for logging)
        
        Uses softmax over UCB scores
        """
        x = self.featurize(state)
        
        scores = []
        for action_idx in range(len(self.actions)):
            A_inv = np.linalg.inv(self.A[action_idx])
            theta = A_inv @ self.b[action_idx]
            
            exploitation = theta.T @ x
            uncertainty = np.sqrt(x.T @ A_inv @ x)
            score = exploitation + self.alpha * uncertainty
            
            scores.append(score)
        
        # Softmax
        scores = np.array(scores)
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        probs = exp_scores / exp_scores.sum()
        
        return probs
    
    def save_state(self) -> Dict:
        """Save bandit state"""
        return {
            'actions': self.actions,
            'alpha': self.alpha,
            'd': self.d,
            'A': {k: v.tolist() for k, v in self.A.items()},
            'b': {k: v.tolist() for k, v in self.b.items()}
        }
    
    def load_state(self, state: Dict):
        """Load bandit state"""
        self.actions = state['actions']
        self.alpha = state['alpha']
        self.d = state['d']
        self.A = {int(k): np.array(v) for k, v in state['A'].items()}
        self.b = {int(k): np.array(v) for k, v in state['b'].items()}


class BanditLogger:
    """Log bandit interactions for offline evaluation"""
    
    def __init__(self):
        self.logs = []
    
    def log(self, 
            state: Dict,
            action_idx: int,
            weights: List[float],
            reward: float,
            propensity: float):
        """
        Log interaction
        
        Args:
            state: Session state
            action_idx: Selected action
            weights: Weight vector used
            reward: Observed reward
            propensity: P(action | state) under logging policy
        """
        self.logs.append({
            'state': state.copy(),
            'action_idx': action_idx,
            'weights': weights.copy(),
            'reward': reward,
            'propensity': propensity
        })
    
    def get_logs(self) -> List[Dict]:
        """Get all logged interactions"""
        return self.logs
    
    def clear(self):
        """Clear logs"""
        self.logs = []
