"""
Counterfactual Evaluation: IPS and SNIPS for off-policy evaluation
"""
import numpy as np
from typing import Dict, List, Tuple


class CounterfactualEvaluator:
    """
    Off-policy evaluation using importance sampling
    
    Estimates new policy performance from logged data
    """
    
    def __init__(self):
        pass
    
    def inverse_propensity_scoring(self,
                                   logged_data: List[Dict],
                                   new_policy) -> Dict:
        """
        Inverse Propensity Scoring (IPS)
        
        Estimates: E[reward | new_policy]
        
        Formula: IPS = (1/N) * Σ (π(a|x) / μ(a|x)) * r
        
        Args:
            logged_data: List of {state, action_idx, reward, propensity}
            new_policy: Policy with get_action_probabilities(state) method
        
        Returns: {
            'ips_reward': float,
            'variance': float,
            'num_samples': int
        }
        """
        if not logged_data:
            return {'ips_reward': 0.0, 'variance': 0.0, 'num_samples': 0}
        
        weighted_rewards = []
        
        for log in logged_data:
            state = log['state']
            action_idx = log['action_idx']
            reward = log['reward']
            p_logging = log['propensity']
            
            # Get new policy probability
            p_new_dist = new_policy.get_action_probabilities(state)
            p_new = p_new_dist[action_idx]
            
            # Importance weight
            weight = p_new / (p_logging + 1e-8)
            
            # Weighted reward
            weighted_reward = weight * reward
            weighted_rewards.append(weighted_reward)
        
        # Estimate
        ips_reward = np.mean(weighted_rewards)
        variance = np.var(weighted_rewards)
        
        return {
            'ips_reward': float(ips_reward),
            'variance': float(variance),
            'std': float(np.sqrt(variance)),
            'num_samples': len(logged_data)
        }
    
    def self_normalized_ips(self,
                           logged_data: List[Dict],
                           new_policy) -> Dict:
        """
        Self-Normalized Inverse Propensity Scoring (SNIPS)
        
        More stable than IPS, especially with high variance
        
        Formula: SNIPS = Σ(w_i * r_i) / Σ(w_i)
        
        Args:
            logged_data: List of {state, action_idx, reward, propensity}
            new_policy: Policy with get_action_probabilities(state) method
        
        Returns: {
            'snips_reward': float,
            'effective_sample_size': float,
            'num_samples': int
        }
        """
        if not logged_data:
            return {'snips_reward': 0.0, 'effective_sample_size': 0.0, 'num_samples': 0}
        
        numerator = 0.0
        denominator = 0.0
        weights = []
        
        for log in logged_data:
            state = log['state']
            action_idx = log['action_idx']
            reward = log['reward']
            p_logging = log['propensity']
            
            # Get new policy probability
            p_new_dist = new_policy.get_action_probabilities(state)
            p_new = p_new_dist[action_idx]
            
            # Importance weight
            weight = p_new / (p_logging + 1e-8)
            weights.append(weight)
            
            numerator += weight * reward
            denominator += weight
        
        # Self-normalized estimate
        snips_reward = numerator / (denominator + 1e-8)
        
        # Effective sample size (measure of variance)
        weights_array = np.array(weights)
        ess = (weights_array.sum() ** 2) / (weights_array ** 2).sum()
        
        return {
            'snips_reward': float(snips_reward),
            'effective_sample_size': float(ess),
            'num_samples': len(logged_data),
            'weight_mean': float(np.mean(weights)),
            'weight_std': float(np.std(weights))
        }
    
    def compare_policies(self,
                        logged_data: List[Dict],
                        policies: Dict[str, any]) -> Dict:
        """
        Compare multiple policies using IPS and SNIPS
        
        Args:
            logged_data: Logged interactions
            policies: {policy_name: policy_object}
        
        Returns: Comparison results
        """
        results = {}
        
        for policy_name, policy in policies.items():
            ips_result = self.inverse_propensity_scoring(logged_data, policy)
            snips_result = self.self_normalized_ips(logged_data, policy)
            
            results[policy_name] = {
                'ips': ips_result,
                'snips': snips_result
            }
        
        return results
    
    def compute_confidence_interval(self,
                                   logged_data: List[Dict],
                                   policy,
                                   confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute confidence interval for IPS estimate
        
        Uses normal approximation
        """
        ips_result = self.inverse_propensity_scoring(logged_data, policy)
        
        mean = ips_result['ips_reward']
        std = ips_result['std']
        n = ips_result['num_samples']
        
        # Z-score for confidence level
        if confidence == 0.95:
            z = 1.96
        elif confidence == 0.99:
            z = 2.576
        else:
            z = 1.96
        
        margin = z * (std / np.sqrt(n))
        
        lower = mean - margin
        upper = mean + margin
        
        return (lower, upper)


class RewardCalculator:
    """Calculate reward from interaction outcomes"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Reward weights (from Phase 0 design)
        self.w_click = config.get('w_click', 0.4)
        self.w_dwell = config.get('w_dwell', 0.3)
        self.w_continue = config.get('w_continue', 0.2)
        self.w_diversity = config.get('w_diversity', 0.1)
    
    def compute_reward(self,
                      clicked: bool,
                      dwell_time: float,
                      session_continues: bool,
                      diversity_gain: float) -> float:
        """
        Compute reward from interaction outcomes
        
        Args:
            clicked: Whether item was clicked
            dwell_time: Time spent (seconds)
            session_continues: Whether user continued session
            diversity_gain: Change in list diversity
        
        Returns: Reward value
        """
        # Normalize components
        click_indicator = 1.0 if clicked else 0.0
        dwell_norm = min(dwell_time / 60.0, 1.0)  # Cap at 1 minute
        continue_indicator = 1.0 if session_continues else 0.0
        diversity_norm = max(0.0, min(diversity_gain, 1.0))
        
        # Weighted sum
        reward = (
            self.w_click * click_indicator +
            self.w_dwell * dwell_norm +
            self.w_continue * continue_indicator +
            self.w_diversity * diversity_norm
        )
        
        return reward
