# Phase 0: Problem Framing & Metrics Definition

## 1. Problem Definition

**System Type:** Session-aware multi-objective decision system for news feed ranking

**Core Objective:** Dynamically rank news articles to maximize long-term business value by balancing competing objectives (engagement, retention, diversity, novelty) based on real-time session context.

**Key Insight:** Ranking is a business decision problem, not just a prediction problem. Optimal trade-offs between objectives vary by session state.

---

## 2. Formal Scoring Function

### Canonical Formulation

```
Score(i, s) = w₁(s) · E(i,s) + w₂(s) · R(i,s) + w₃(s) · D(i,L) + w₄(s) · N(i)
```

**Where:**
- `i`: candidate article
- `s`: current session state
- `L`: current ranked list (for diversity calculation)
- `w(s) = [w₁, w₂, w₃, w₄]`: session-dependent weight vector

**Constraints:**
- `wᵢ ≥ 0` for all i
- `Σwᵢ = 1` (normalized weights)

---

## 3. Component Definitions

### 3.1 Engagement Score: E(i, s)

**Purpose:** Capture immediate user interaction likelihood

**Formula:**
```
E(i, s) = α · CTR(i, s) + (1 - α) · DwellNorm(i, s)
```

**Components:**
- `CTR(i, s)`: Predicted click probability given user, item, and session context
  - Model: `P(click | user_embedding, item_embedding, session_features)`
- `DwellNorm(i, s)`: Normalized expected dwell time
  - Normalization: `log(1 + dwell_seconds) / log(1 + max_dwell_threshold)`
  - Prevents long-tail spikes from dominating

**Hyperparameter:**
- `α = 0.6` (initial value, tunable)

**Range:** [0, 1]

---

### 3.2 Retention Proxy: R(i, s)

**Purpose:** Estimate likelihood of future user return

**Formula (Phase 0 - Simple):**
```
R(i, s) = P(session_continues | state, item)
```

**Definition:**
- Binary target: Did user continue session after this interaction?
- Threshold: Session continues if next interaction occurs within 5 minutes

**Model Input:**
- Session features: length, avg_dwell, click_entropy
- Item features: category, quality_score
- Interaction features: position, dwell_time

**Future Upgrade Path:**
- Phase 3+: Model next-day return probability
- Requires longer-term behavioral data

**Range:** [0, 1]

---

### 3.3 Diversity Score: D(i, L)

**Purpose:** Reduce redundancy and filter bubble effects

**Formula (Similarity-Based):**
```
D(i, L) = 1 - avg_cosine_similarity(emb(i), emb(j) for j in L)
```

**Implementation:**
- `emb(i)`: Article embedding (from two-tower model or BERT)
- `L`: Top-K articles already in ranked list (K=10 for efficiency)
- Compute average pairwise similarity between candidate and existing items

**Alternative (Category-Based - Fallback):**
```
D(i, L) = entropy(category_distribution(L ∪ {i}))
```

**Range:** [0, 1]
- 0 = highly similar to existing items
- 1 = maximally diverse

---

### 3.4 Novelty Score: N(i)

**Purpose:** Balance freshness and exploration of less-popular content

**Formula:**
```
N(i) = β · Freshness(i) + (1 - β) · Unpopularity(i)
```

**Components:**
- `Freshness(i) = exp(-λ · age_hours)`
  - `λ = 0.1` (decay rate, tunable)
  - Recent articles score higher
  
- `Unpopularity(i) = 1 / (1 + log(1 + global_click_count))`
  - Inverse popularity with log smoothing
  - Prevents over-penalizing moderately popular items

**Hyperparameter:**
- `β = 0.7` (favor freshness over unpopularity)

**Range:** [0, 1]

---

## 4. Session State Definition

### Core Features

```python
session_state = {
    # Interaction metrics
    "session_length": int,              # Number of interactions so far
    "avg_dwell_time": float,            # Mean dwell time (seconds)
    "total_session_time": float,        # Time since session start (minutes)
    
    # Behavioral signals
    "click_entropy": float,             # Diversity of clicked categories
    "skip_rate": float,                 # Fraction of impressions without click
    "recent_categories": List[str],     # Last 5 clicked categories
    
    # Temporal context
    "time_of_day": int,                 # Hour (0-23)
    "day_of_week": int,                 # 0=Monday, 6=Sunday
    
    # Derived indicators
    "fatigue_score": float,             # Composite fatigue indicator
}
```

### Derived Feature: Fatigue Score

```
fatigue_score = 0.5 · (1 - normalized_dwell) + 0.5 · skip_rate
```

**Interpretation:**
- High fatigue → boost novelty/diversity
- Low fatigue → maintain engagement focus

---

## 5. Weight Function Strategy

### Hybrid Approach

**Base Weights (Rule-Based):**

| Session Phase | Condition | w₁ (Engage) | w₂ (Retain) | w₃ (Diverse) | w₄ (Novel) |
|--------------|-----------|-------------|-------------|--------------|------------|
| Early | length < 3 | 0.25 | 0.20 | 0.35 | 0.20 |
| Mid (engaged) | avg_dwell > 30s | 0.50 | 0.25 | 0.15 | 0.10 |
| Late (fatigued) | fatigue > 0.6 | 0.20 | 0.30 | 0.25 | 0.25 |
| Default | otherwise | 0.40 | 0.25 | 0.20 | 0.15 |

**Learned Refinement (Contextual Bandit - Phase 6):**
- State: `session_state` features
- Action: Weight adjustment Δw
- Policy: Neural network or LinUCB
- Exploration: ε-greedy or UCB

---

## 6. Reward Function

**Purpose:** Single metric for bandit optimization

**Formula:**
```
reward = 0.4 · click_indicator +
         0.3 · normalized_dwell +
         0.2 · session_continues +
         0.1 · diversity_gain
```

**Component Definitions:**
- `click_indicator`: Binary (1 if clicked, 0 otherwise)
- `normalized_dwell`: `min(dwell_seconds / 60, 1.0)`
- `session_continues`: Binary (1 if user continues within 5 min)
- `diversity_gain`: Change in list diversity after adding item

**Normalization:** All components scaled to [0, 1]

**Rationale:**
- Weights reflect business priorities
- Immediate feedback (click, dwell) weighted higher than delayed (retention)
- Diversity prevents filter bubbles

---

## 7. Evaluation Metrics

### Offline Metrics (Model Quality)

**Ranking Metrics:**
- NDCG@10, NDCG@20
- Recall@10, Recall@20
- MRR (Mean Reciprocal Rank)

**Business Proxy Metrics:**
- Predicted CTR (AUC, log-loss)
- Predicted dwell time (MAE, RMSE)
- Retention prediction accuracy

### Online Simulation Metrics (System Quality)

**Engagement:**
- Average CTR per session
- Total dwell time per session
- Click-through depth (how far users scroll)

**Retention:**
- Session continuation rate
- Average session length
- Simulated next-day return rate

**Structural Quality:**
- Intra-list diversity (avg pairwise similarity)
- Category entropy
- Novelty distribution (% fresh vs. popular)

**Multi-Objective Score:**
```
MO_Score = 0.4·CTR + 0.3·AvgDwell + 0.2·SessionContinues + 0.1·Diversity
```

---

## 8. Baseline Definition

### Baseline 1: Engagement-Only
```
Score(i, s) = E(i, s)
```
Pure CTR + dwell optimization (industry standard)

### Baseline 2: Fixed Weights
```
Score(i, s) = 0.4·E + 0.25·R + 0.2·D + 0.15·N
```
Static multi-objective combination (no session adaptation)

### Baseline 3: Popularity
```
Score(i) = global_click_count
```
Non-personalized ranking (cold-start fallback)

**Success Criteria:**
- Adaptive system must outperform all baselines on multi-objective score
- Acceptable trade-offs: -5% CTR for +15% retention or +20% diversity

---

## 9. Key Assumptions

1. **Session Approximation:** User behavior within a session can be modeled from interaction logs (clicks, dwell, skips)

2. **Dwell Time Signal:** Dwell time correlates with content quality and satisfaction (not just article length)

3. **Diversity-Retention Link:** Increased diversity reduces filter bubble fatigue and improves long-term retention (hypothesis to validate)

4. **Weight Adaptability:** Optimal objective weights vary significantly by session state (early vs. late, engaged vs. fatigued)

5. **Reward Decomposition:** Business value can be approximated as a weighted combination of engagement, retention, and structural metrics

6. **Bandit Feasibility:** Contextual bandit can learn effective weight policies within reasonable sample complexity

7. **Logged Data Sufficiency:** MIND dataset + simulated sessions provide sufficient signal for offline development and counterfactual evaluation

---

## 10. Risk Mitigation

| Risk | Mitigation Strategy |
|------|---------------------|
| Cold-start (new users) | Fallback to popularity + diversity baseline |
| Popularity bias | Explicit novelty term + exploration in bandit |
| Feedback loops | Diversity enforcement + bandit exploration |
| Latency constraints | Pre-compute embeddings, cache scores, limit list size |
| Reward sparsity | Use immediate proxies (dwell) for delayed signals (retention) |

---

## 11. Success Criteria (Phase 0)

✅ All components mathematically defined
✅ Scoring function is computable and differentiable
✅ Metrics are measurable from available data
✅ Baselines are implementable
✅ Assumptions are explicit and testable

**Next Phase:** Data exploration and feature engineering (Phase 1)
