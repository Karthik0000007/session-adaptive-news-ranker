# Production Readiness Guide

## Phase 11: Failure Modes, System Design & Production Considerations

This document outlines production deployment considerations, failure modes, monitoring strategies, and operational best practices for the Session-Adaptive News Ranker.

---

## Table of Contents

1. [Failure Modes & Mitigations](#failure-modes--mitigations)
2. [System Constraints](#system-constraints)
3. [Monitoring & Observability](#monitoring--observability)
4. [Deployment Strategy](#deployment-strategy)
5. [Retraining Pipeline](#retraining-pipeline)
6. [Security & Privacy](#security--privacy)
7. [Operational Runbook](#operational-runbook)

---

## Failure Modes & Mitigations

### 1. Cold Start (New User / New Article)

**Problem:**
- No user history → poor personalization
- New articles have no engagement data

**Detection:**
- User history length = 0
- Article age < 1 hour with no interactions

**Mitigation:**
```python
if user_history is empty:
    # Fallback strategy
    return popular_articles + diverse_categories + fresh_content
    
    # Use exploration-heavy weights
    weights = [0.25, 0.20, 0.35, 0.20]  # High diversity/novelty
```

**Better Solutions:**
- Popularity prior with category-level personalization
- Content-based similarity for new articles
- Exploration quota (force 20% new content)

---

### 2. Feedback Loop Bias (Filter Bubble)

**Problem:**
- Model keeps showing similar content
- User trapped in narrow interest space
- Low entropy in recommendations

**Detection:**
```python
if click_entropy < 0.5:
    alert("Low diversity detected")

if repeated_categories > 0.7:
    alert("Filter bubble risk")
```

**Mitigation:**
- Increase diversity weight dynamically
- Inject novelty items (force exploration)
- Diversity quota per session
- Category cap (max 40% from single category)

**Implementation:**
```python
def apply_diversity_boost(weights, entropy):
    if entropy < 0.5:
        # Boost diversity and novelty
        weights[2] *= 1.5  # Diversity
        weights[3] *= 1.3  # Novelty
        # Renormalize
        weights = weights / sum(weights)
    return weights
```

---

### 3. Popularity Bias

**Problem:**
- Popular items dominate rankings
- Long-tail content never surfaces
- Reduced discovery

**Detection:**
- Top 10% articles get >80% impressions
- Gini coefficient > 0.8

**Mitigation:**
```python
novelty_score = 0.7 * freshness + 0.3 * (1 / log(popularity + 1))

# Cap popularity influence
popularity_weight = min(popularity_score, 0.7)
```

**Strategies:**
- Inverse popularity weighting
- Time-decay for popular items
- Exploration slots for long-tail

---

### 4. Bandit Over-Exploitation

**Problem:**
- Policy stops exploring too early
- Stuck in local optimum
- No adaptation to changing user behavior

**Detection:**
```python
if action_entropy < 0.3:
    alert("Bandit over-exploiting")

if single_action_frequency > 0.8:
    alert("No exploration")
```

**Mitigation:**
- Increase exploration parameter α
- Epsilon-greedy fallback (ε = 0.1)
- Periodic exploration phases
- Action diversity constraint

**Implementation:**
```python
def select_action_with_exploration(state, epsilon=0.1):
    if random() < epsilon:
        return random_action()  # Explore
    else:
        return bandit.select_action(state)  # Exploit
```

---

### 5. Data Drift

**Problem:**
- User behavior changes over time
- Seasonal patterns
- Model performance degrades

**Detection:**
- CTR drops >10% week-over-week
- Dwell time distribution shift (KL divergence)
- Session length decrease

**Mitigation:**
- Sliding window training (last 30 days)
- Periodic retraining (weekly)
- Online learning for bandit
- A/B test new models before deployment

---

### 6. Latency Spikes

**Problem:**
- P99 latency >150ms
- User experience degradation
- Timeout errors

**Detection:**
- Monitor P50, P95, P99 latency
- Alert if P99 >200ms

**Mitigation:**
- Caching layer (Redis)
- Precompute embeddings
- Fallback to cached results
- Circuit breaker pattern
- Request timeout (150ms)

**Implementation:**
```python
@timeout(150)  # milliseconds
def rank_with_timeout(user_id, state):
    try:
        return rank(user_id, state)
    except TimeoutError:
        return cached_fallback(user_id)
```

---

### 7. Model Serving Failure

**Problem:**
- Model loading error
- Prediction failure
- Service crash

**Detection:**
- Health check endpoint fails
- Error rate >1%

**Mitigation:**
- Graceful degradation
- Fallback to rule-based ranking
- Model versioning
- Blue-green deployment

**Fallback Chain:**
```
Bandit → Rule-based → Fixed weights → Popular items
```

---

## System Constraints

### Latency Budget

| Component | Target | P99 Limit |
|---|---|---|
| Retrieval | <50ms | <80ms |
| Ranking | <50ms | <80ms |
| Decision Layer | <30ms | <50ms |
| **Total** | **<130ms** | **<150ms** |

**Trade-offs:**
- Deeper models → Higher latency
- More candidates → Slower ranking
- Real-time features → Increased latency

**Optimization Strategies:**
- Approximate nearest neighbor (FAISS IVF)
- Model quantization
- Batch prediction
- Feature caching

---

### Scalability

**Current Capacity:**
- 100K articles
- 10K concurrent users
- 1M requests/day

**Scaling Strategies:**

1. **Horizontal Scaling:**
   - Load balancer (NGINX)
   - Multiple API instances
   - Stateless design

2. **Caching:**
   - Redis for session state
   - CDN for static content
   - Result caching (5 min TTL)

3. **Database:**
   - Read replicas
   - Sharding by user_id
   - Connection pooling

4. **FAISS Index:**
   - IVF index for >1M articles
   - GPU acceleration
   - Distributed index

---

### Consistency vs Freshness

**Trade-off:**
- Fresh data → Slower, more load
- Cached data → Faster, potentially stale

**Strategy:**
```python
# Hybrid approach
if cache_age < 5_minutes:
    return cached_result
else:
    result = compute_fresh()
    cache.set(result, ttl=300)
    return result
```

**Refresh Policies:**
- Session state: Real-time
- User embeddings: 1 hour
- Article embeddings: 6 hours
- Popular items: 1 hour

---

## Monitoring & Observability

### Model Metrics

**Business Metrics (Real-time):**
```python
metrics = {
    'ctr': CTR,
    'avg_dwell_time': seconds,
    'session_length': interactions,
    'session_continuation_rate': percentage,
    'diversity_score': 0-1,
    'novelty_score': 0-1
}
```

**Frequency:** Log every request, aggregate hourly

**Alerts:**
- CTR drops >10% (1 hour window)
- Avg dwell <20s
- Session length <3

---

### System Metrics

**Infrastructure:**
```python
system_metrics = {
    'api_latency_p50': ms,
    'api_latency_p95': ms,
    'api_latency_p99': ms,
    'error_rate': percentage,
    'requests_per_second': count,
    'cache_hit_rate': percentage,
    'redis_memory_usage': MB,
    'cpu_usage': percentage,
    'memory_usage': MB
}
```

**Alerts:**
- P99 latency >150ms
- Error rate >1%
- Cache hit rate <70%
- CPU >80%

---

### Distribution Monitoring

**Detect Drift:**
```python
# Compare distributions week-over-week
kl_divergence(current_week, previous_week)

# Monitor:
- session_length_distribution
- dwell_time_distribution
- click_entropy_distribution
- category_distribution
```

**Alert Thresholds:**
- KL divergence >0.3
- Mean shift >20%
- Variance change >50%

---

### Bandit Metrics

```python
bandit_metrics = {
    'reward_mean': float,
    'reward_std': float,
    'action_entropy': 0-2,  # Higher = more exploration
    'ips_estimate': float,
    'snips_estimate': float,
    'effective_sample_size': int
}
```

**Alerts:**
- Reward drops >15%
- Action entropy <0.3 (over-exploitation)

---

### Logging

**What to Log:**
```python
log_entry = {
    'timestamp': datetime,
    'user_id': str,
    'session_id': str,
    'state': dict,
    'action': list,  # weights
    'ranked_list': list,
    'user_response': dict,
    'reward': float,
    'latency_ms': float,
    'strategy': str
}
```

**Storage:**
- Real-time: Redis stream
- Batch: S3 / Cloud Storage
- Analytics: Data warehouse

---

## Deployment Strategy

### Canary Release

```
1. Deploy to 5% of traffic
2. Monitor for 1 hour
3. If metrics stable → 25%
4. Monitor for 2 hours
5. If metrics stable → 100%
```

**Rollback Criteria:**
- Error rate >2%
- P99 latency >200ms
- CTR drops >15%

---

### Blue-Green Deployment

```
Blue (current) ← 100% traffic
Green (new) ← 0% traffic

Switch:
Blue ← 0% traffic
Green ← 100% traffic

Keep Blue for 24h (rollback)
```

---

### Feature Flags

```python
if feature_flag('use_bandit'):
    strategy = 'bandit'
else:
    strategy = 'rule_based'
```

**Use Cases:**
- Gradual rollout
- A/B testing
- Emergency disable

---

## Retraining Pipeline

### Offline Loop

```
Logs → Feature Pipeline → Model Training → Validation → Deployment
```

**Frequency:**

| Component | Frequency | Reason |
|---|---|---|
| Bandit | Real-time | Online learning |
| Ranking Models | Daily | Capture trends |
| Retrieval | Weekly | Stable embeddings |
| Popular Items | Hourly | Fast-changing |

---

### Training Pipeline

```python
# Daily at 2 AM
def retrain_ranking_models():
    # 1. Extract logs (last 30 days)
    data = load_logs(days=30)
    
    # 2. Feature engineering
    features = extract_features(data)
    
    # 3. Train models
    ctr_model = train_ctr(features)
    dwell_model = train_dwell(features)
    retention_model = train_retention(features)
    
    # 4. Validate
    metrics = validate(models, test_data)
    
    # 5. Deploy if better
    if metrics['auc'] > current_model_auc:
        deploy(models)
    else:
        alert("New model underperforms")
```

---

### Online Learning (Bandit)

```python
# Real-time updates
def update_bandit(state, action, reward):
    bandit.update(state, action, reward)
    
    # Persist every 1000 updates
    if update_count % 1000 == 0:
        save_bandit_state()
```

---

## Security & Privacy

### Data Privacy

**User Data:**
- Session-based modeling (no long-term profiles)
- Anonymized user IDs
- No PII in logs

**Compliance:**
- GDPR: Right to deletion
- APPI (Japan): Data minimization
- CCPA: Opt-out mechanism

---

### API Security

```python
# Rate limiting
@limiter.limit("100/minute")
def rank_endpoint():
    ...

# Authentication
@require_api_key
def rank_endpoint():
    ...

# Input validation
def validate_request(request):
    assert 0 <= request.k <= 100
    assert request.user_id.isalnum()
```

---

## Operational Runbook

### Common Issues

#### Issue: High Latency

**Symptoms:** P99 >200ms

**Diagnosis:**
```bash
# Check API latency
curl /metrics | grep latency

# Check Redis
redis-cli info | grep used_memory

# Check model loading
tail -f logs/app.log | grep "model load"
```

**Resolution:**
1. Check cache hit rate
2. Restart Redis if memory >80%
3. Scale API instances
4. Enable result caching

---

#### Issue: Low CTR

**Symptoms:** CTR drops >10%

**Diagnosis:**
```python
# Check diversity
SELECT AVG(diversity_score) FROM logs
WHERE timestamp > NOW() - INTERVAL 1 HOUR

# Check action distribution
SELECT action, COUNT(*) FROM bandit_logs
GROUP BY action
```

**Resolution:**
1. Increase exploration (α)
2. Boost diversity weight
3. Check for filter bubbles
4. Retrain models

---

#### Issue: Model Serving Error

**Symptoms:** Error rate >1%

**Diagnosis:**
```bash
# Check logs
tail -f logs/error.log

# Check model files
ls -lh data/processed/ranking_models/
```

**Resolution:**
1. Rollback to previous version
2. Check model file integrity
3. Restart service
4. Use fallback strategy

---

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Redis health
redis-cli ping

# Model loaded
curl http://localhost:8000/metrics | grep models_loaded
```

---

### Emergency Procedures

**Scenario: Complete System Failure**

1. Enable fallback mode (popular items)
2. Alert on-call engineer
3. Check logs for root cause
4. Rollback to last stable version
5. Post-mortem analysis

---

## Trade-off Thinking

### Key Trade-offs

1. **Engagement vs Diversity**
   - High CTR → Low diversity → Filter bubbles
   - High diversity → Lower CTR → Better long-term retention

2. **Exploration vs Exploitation**
   - More exploration → Better discovery → Lower short-term metrics
   - More exploitation → Higher immediate metrics → Missed opportunities

3. **Latency vs Accuracy**
   - Complex models → Better predictions → Higher latency
   - Simple models → Faster → Lower accuracy

4. **Freshness vs Stability**
   - Frequent retraining → Adapts quickly → Risk of instability
   - Infrequent retraining → Stable → Misses trends

---

## Production Checklist

- [ ] All models trained and validated
- [ ] API deployed with health checks
- [ ] Redis configured and tested
- [ ] Monitoring dashboards set up
- [ ] Alerts configured
- [ ] Logging pipeline active
- [ ] Fallback strategies tested
- [ ] Load testing completed
- [ ] Security review passed
- [ ] Documentation complete
- [ ] On-call rotation established
- [ ] Runbook reviewed

---

## Conclusion

This system is designed for production deployment with:
- Graceful degradation
- Comprehensive monitoring
- Automated retraining
- Security best practices
- Operational runbooks

**Key Principle:** Always have a fallback. Never let the system fail completely.

---

**Last Updated:** 2026-04-07  
**Version:** 1.0
