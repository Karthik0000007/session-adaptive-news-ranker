# Session-Adaptive Multi-Objective News Feed Optimizer

A production-ready personalized news feed ranking system that treats ranking as a business decision problem, dynamically balancing engagement, retention, diversity, and novelty based on real-time session context.

## Project Overview

This system uses a hybrid dynamic weighting layer powered by a contextual bandit to learn optimal trade-offs between competing business objectives, conditioned on session state. The feed adapts intelligently (e.g., prioritize diversity early to reduce bounce rate, shift to engagement as dwell time rises, inject novelty later to fight churn).

## Architecture

```
Data → Candidate Generation → Base Ranking → Decision Layer → Serving
         (Two-Tower + FAISS)   (LightGBM/DL)   (Contextual Bandit)
```

## Current Status

### ✅ Problem Framing & Metrics Definition
- Formal scoring function defined
- All objective components (E, R, D, N) mathematically specified
- Session state schema designed
- Evaluation metrics established

### ✅ Dataset & Session Simulation
- MIND dataset integration
- Session construction from impressions
- Behavioral signal simulation (dwell time, fatigue)
- Feature extraction pipeline

### ✅ Candidate Generation (Retrieval System)
- Two-tower architecture (item + user encoders)
- FAISS index for fast similarity search
- Session-aware user embeddings
- Retrieval evaluation (Recall@K, Hit Rate@K, MRR)

### ✅ Base Ranking Models
- LightGBM models for CTR, dwell time, and retention
- Comprehensive feature engineering (user, item, session, interaction)
- Probability calibration (Platt scaling)
- Time-based train/validation split

### ✅ Multi-Objective Decision Layer
- Objective scoring (engagement, retention, diversity, novelty)
- Greedy diversity-aware reranking
- Rule-based adaptive weighting
- Session-aware weight adjustment

### ✅ Contextual Bandit & Counterfactual Evaluation
- LinUCB with discrete action space (4 weight strategies)
- IPS and SNIPS counterfactual evaluation
- Reward calculation for multi-objective optimization
- Propensity logging for off-policy evaluation

### ✅ A/B Testing Simulation
- User behavior simulation (click, dwell, continuation)
- Strategy comparison (baseline, rule-based, bandit, engagement-only)
- Realistic session dynamics with fatigue modeling
- Comprehensive metrics tracking

### ✅ Production Serving
- FastAPI-based REST API
- Redis session management
- Result caching with TTL
- Cold-start fallback logic
- Health checks and metrics endpoints

### ✅ Visualization Dashboard
- Interactive Streamlit dashboard
- Trade-off analysis (CTR vs Diversity)
- Session evolution visualization
- Strategy comparison radar charts
- Bandit learning curves

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Setup
1. Download MIND dataset (small or large)
2. Place `behaviors.tsv` and `news.tsv` in `data/raw/`

### Run Data Pipeline
```bash
python pipeline.py
```

This will:
- Parse MIND impressions into interaction events
- Build user sessions with 30-minute gap threshold
- Simulate dwell time and fatigue signals
- Extract session state features
- Generate training samples with negative sampling

### Train Retrieval System
```bash
python train_retrieval.py
```

This will:
- Train two-tower model (item + user encoders)
- Precompute item embeddings
- Build FAISS index for fast retrieval
- Evaluate retrieval performance (Recall@K, Hit Rate@K)

### Train Ranking Models
```bash
python train_ranker.py
```

This will:
- Build feature engineering pipeline (user profiles, item stats)
- Train CTR, dwell, and retention models (LightGBM)
- Calibrate probability predictions
- Evaluate model performance (AUC, RMSE, LogLoss)

### Evaluate Decision Layer
```bash
python evaluate_decision_layer.py
```

This will:
- Test multi-objective scoring with adaptive weights
- Compare fixed vs rule-based weight strategies
- Measure engagement, retention, diversity, novelty

### Train Contextual Bandit
```bash
python train_bandit.py
```

This will:
- Train LinUCB contextual bandit
- Learn optimal weight selection policy
- Evaluate with IPS/SNIPS counterfactual methods
- Compare against baseline strategies

### Run A/B Test Simulation
```bash
python simulate_ab_test.py
```

This will:
- Simulate user behavior across 1000+ sessions
- Compare all strategies (baseline, rule-based, bandit, engagement-only)
- Generate comprehensive comparison metrics
- Save results for visualization

### Start Production Server
```bash
# Start Redis (required)
redis-server

# Start API server
python serve.py
```

API will be available at `http://localhost:8000`

Example request:
```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "U12345",
    "k": 20,
    "strategy": "bandit"
  }'
```

### Launch Visualization Dashboard
```bash
streamlit run dashboard.py
```

Dashboard will open at `http://localhost:8501`

Features:
- Trade-off curves (CTR vs Diversity)
- Session evolution visualization
- Strategy comparison
- Bandit learning dynamics

### Configuration
Edit `config.yaml` to adjust:
- Session gap threshold
- Dwell time distributions
- Number of negative samples
- Feature extraction parameters

## Project Structure

```
session-adaptive-news-ranker/
├── config.yaml                 # Configuration
├── pipeline.py                 # Data processing pipeline
├── train_retrieval.py          # Retrieval system training
├── train_ranker.py             # Ranking models training
├── evaluate_decision_layer.py  # Decision layer evaluation
├── train_bandit.py             # Contextual bandit training
├── simulate_ab_test.py         # A/B testing simulation
├── serve.py                    # Production API server
├── dashboard.py                # Streamlit visualization
├── requirements.txt
├── DESIGN.md                   # Formal problem definition
├── ARCHITECTURE.md             # Technical architecture
├── docs/
│   ├── data_pipeline_guide.md  # Data processing guide
│   ├── retrieval_guide.md      # Retrieval system guide
│   ├── ranking_guide.md        # Ranking models guide
│   └── production_guide.md     # Production deployment guide
├── src/
│   ├── data_loader.py          # MIND dataset loading
│   ├── session_builder.py      # Session construction
│   ├── signal_simulator.py     # Behavioral signal simulation
│   ├── feature_extractor.py    # Session state features
│   ├── negative_sampler.py     # Negative sampling
│   ├── item_encoder.py         # Article encoder (TF-IDF + categories)
│   ├── user_encoder.py         # Session-aware user encoder
│   ├── two_tower_model.py      # Two-tower orchestration
│   ├── faiss_index.py          # FAISS index for retrieval
│   ├── retrieval_system.py     # End-to-end retrieval API
│   ├── retrieval_evaluator.py  # Retrieval evaluation metrics
│   ├── base_ranker.py          # LightGBM ranking model
│   ├── ranking_features.py     # Feature engineering pipeline
│   ├── ranking_system.py       # Multi-objective ranking orchestration
│   ├── objective_scorer.py     # Multi-objective scoring
│   ├── weight_adapter.py       # Rule-based weight adaptation
│   ├── decision_layer.py       # Decision layer integration
│   ├── contextual_bandit.py    # LinUCB implementation
│   ├── counterfactual_evaluator.py  # IPS/SNIPS evaluation
│   └── session_manager.py      # Redis session management
├── data/
│   ├── raw/                    # MIND dataset files
│   └── processed/              # Generated sessions, features, models
└── notebooks/                  # Analysis notebooks (coming soon)
```

## Key Features

### Session-Aware Ranking
- Dynamic weight adjustment based on session state
- Fatigue detection and response
- Temporal context integration

### Multi-Objective Optimization
- Engagement (CTR + dwell time)
- Retention (session continuation)
- Diversity (category entropy)
- Novelty (freshness + unpopularity)

### Production-Ready Design
- Low-latency serving architecture
- Failure mode handling (cold-start, popularity bias)
- Counterfactual evaluation (IPS/SNIPS)

## Roadmap

- [x] Problem Framing & Design
- [x] Dataset & Session Simulation
- [x] Candidate Generation (Retrieval)
- [x] Base Ranking Models
- [x] Multi-Objective Decision Layer
- [x] Contextual Bandit Integration
- [x] Counterfactual Evaluation (IPS/SNIPS)
- [x] A/B Testing Simulation
- [x] Production Serving Infrastructure
- [x] Visualization Dashboard
- [x] Production Readiness Documentation

## Key Deliverables

### Phase 1-3: Core ML Pipeline
- Data processing with session construction
- Two-tower retrieval system with FAISS
- LightGBM ranking models (CTR, dwell, retention)

### Phase 4-5: Decision Layer
- Multi-objective scoring (E·R·D·N)
- Rule-based adaptive weighting
- Greedy diversity-aware reranking

### Phase 6-7: Bandit & Evaluation
- LinUCB contextual bandit
- IPS/SNIPS counterfactual evaluation
- Off-policy learning from logged data

### Phase 8-9: Simulation & Serving
- Realistic user behavior simulation
- FastAPI production endpoint
- Redis session management
- Caching and fallback strategies

### Phase 10: Visualization
- Interactive Streamlit dashboard
- Trade-off analysis
- Strategy comparison
- Learning dynamics visualization

### Phase 11: Production Readiness
- Failure mode analysis
- Monitoring and alerting strategies
- Retraining pipeline design
- Security and privacy considerations

## Dataset

Using [Microsoft MIND](https://msnews.github.io/) (Microsoft News Dataset):
- 1M+ user interactions
- 160K+ news articles
- Rich metadata (categories, entities, abstracts)

## License

MIT

## Contact

For questions or collaboration, please open an issue.
