"""
Train Contextual Bandit with Counterfactual Evaluation

Learns optimal weight selection policy from simulated sessions
"""
import yaml
import pickle
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.contextual_bandit import LinUCB, BanditLogger
from src.counterfactual_evaluator import CounterfactualEvaluator, RewardCalculator
from src.weight_adapter import WeightAdapter


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add bandit config
    config['bandit'] = {
        'alpha': 0.5  # Exploration parameter
    }
    
    config['reward'] = {
        'w_click': 0.4,
        'w_dwell': 0.3,
        'w_continue': 0.2,
        'w_diversity': 0.1
    }
    
    return config


def load_sessions(processed_dir: str):
    """Load Phase 1 sessions"""
    with open(Path(processed_dir) / 'sessions.pkl', 'rb') as f:
        return pickle.load(f)


def simulate_logging_policy(session_state: Dict) -> Tuple[int, float]:
    """
    Simulate a logging policy (random with slight bias)
    
    Returns: (action_idx, propensity)
    """
    # Simple random policy with uniform probabilities
    num_actions = 4
    action_idx = np.random.randint(0, num_actions)
    propensity = 1.0 / num_actions
    
    return action_idx, propensity


def main():
    print("=" * 60)
    print("Phase 6 & 7: Contextual Bandit + Counterfactual Evaluation")
    print("=" * 60)
    
    # Load config
    config = load_config()
    processed_dir = Path(config['data']['processed_dir'])
    
    # Load sessions
    print("\n[1/5] Loading session data...")
    sessions = load_sessions(processed_dir)
    print(f"  - Loaded {len(sessions)} users")
    
    # Initialize components
    print("\n[2/5] Initializing bandit and evaluator...")
    bandit = LinUCB(config['bandit'])
    logger = BanditLogger()
    reward_calculator = RewardCalculator(config['reward'])
    evaluator = CounterfactualEvaluator()
    
    # Baseline policies for comparison
    rule_based_adapter = WeightAdapter(config.get('weighting', {}))
    
    # Collect training data with logging policy
    print("\n[3/5] Collecting logged data with random policy...")
    
    num_train_sessions = min(500, len(sessions))
    train_users = list(sessions.keys())[:num_train_sessions]
    
    for user_id in tqdm(train_users, desc="  Logging"):
        user_sessions = sessions[user_id]
        
        for session in user_sessions[:1]:  # First session per user
            if len(session) < 2:
                continue
            
            # Build session state
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
            
            # Logging policy selects action
            action_idx, propensity = simulate_logging_policy(session_state)
            weights = bandit.get_action_weights(action_idx)
            
            # Simulate reward (simplified)
            clicked = any(e['clicked'] for e in session)
            avg_dwell = session_state['avg_dwell_time']
            session_continues = len(session) > 3
            diversity_gain = 0.5
            
            reward = reward_calculator.compute_reward(
                clicked=clicked,
                dwell_time=avg_dwell,
                session_continues=session_continues,
                diversity_gain=diversity_gain
            )
            
            # Log interaction
            logger.log(session_state, action_idx, weights, reward, propensity)
    
    logged_data = logger.get_logs()
    print(f"  - Collected {len(logged_data)} logged interactions")
    print(f"  - Average reward: {np.mean([log['reward'] for log in logged_data]):.4f}")
    
    # Train bandit on logged data
    print("\n[4/5] Training LinUCB bandit...")
    
    for log in tqdm(logged_data, desc="  Training"):
        state = log['state']
        action_idx = log['action_idx']
        reward = log['reward']
        
        bandit.update(state, action_idx, reward)
    
    print("  - Bandit training complete")
    
    # Counterfactual evaluation
    print("\n[5/5] Counterfactual evaluation...")
    
    # Create baseline policy wrapper
    class RuleBasedPolicy:
        def __init__(self, adapter, bandit_actions):
            self.adapter = adapter
            self.bandit_actions = bandit_actions
        
        def get_action_probabilities(self, state):
            # Get rule-based weights
            weights = self.adapter.get_weights(state, smooth=False)
            
            # Find closest action
            distances = [
                np.linalg.norm(np.array(weights) - np.array(action))
                for action in self.bandit_actions
            ]
            closest_idx = np.argmin(distances)
            
            # Deterministic policy (all probability on closest action)
            probs = np.zeros(len(self.bandit_actions))
            probs[closest_idx] = 1.0
            
            return probs
    
    rule_policy = RuleBasedPolicy(rule_based_adapter, bandit.actions)
    
    # Evaluate policies
    policies = {
        'LinUCB Bandit': bandit,
        'Rule-Based': rule_policy
    }
    
    comparison = evaluator.compare_policies(logged_data, policies)
    
    print("\n" + "=" * 60)
    print("Counterfactual Evaluation Results")
    print("=" * 60)
    
    for policy_name, results in comparison.items():
        print(f"\n{policy_name}:")
        print(f"  IPS Reward:  {results['ips']['ips_reward']:.4f} "
              f"(±{results['ips']['std']:.4f})")
        print(f"  SNIPS Reward: {results['snips']['snips_reward']:.4f}")
        print(f"  Effective Sample Size: {results['snips']['effective_sample_size']:.1f}")
    
    # Compute confidence intervals
    print("\n" + "=" * 60)
    print("95% Confidence Intervals (IPS)")
    print("=" * 60)
    
    for policy_name, policy in policies.items():
        lower, upper = evaluator.compute_confidence_interval(logged_data, policy)
        print(f"{policy_name:20s}: [{lower:.4f}, {upper:.4f}]")
    
    # Save bandit
    output_dir = processed_dir / 'bandit_model'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bandit_state = bandit.save_state()
    with open(output_dir / 'bandit_state.pkl', 'wb') as f:
        pickle.dump(bandit_state, f)
    
    # Save evaluation results
    eval_results = {
        'comparison': {
            k: {
                'ips_reward': v['ips']['ips_reward'],
                'ips_std': v['ips']['std'],
                'snips_reward': v['snips']['snips_reward'],
                'ess': v['snips']['effective_sample_size']
            }
            for k, v in comparison.items()
        },
        'num_logged_interactions': len(logged_data),
        'config': config['bandit']
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nBandit model saved to: {output_dir}")
    
    print("\n" + "=" * 60)
    print("Phase 6 & 7 Complete!")
    print("=" * 60)
    print("\nKey Achievements:")
    print("- Trained LinUCB contextual bandit for weight selection")
    print("- Implemented IPS and SNIPS for off-policy evaluation")
    print("- Compared learned policy vs rule-based baseline")
    print("- Demonstrated counterfactual evaluation without deployment")


if __name__ == "__main__":
    main()
