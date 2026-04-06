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

### ✅ Phase 0: Problem Framing & Metrics Definition
- Formal scoring function defined
- All objective components (E, R, D, N) mathematically specified
- Session state schema designed
- Evaluation metrics established

### ✅ Phase 1: Dataset & Session Simulation
- MIND dataset integration
- Session construction from impressions
- Behavioral signal simulation (dwell time, fatigue)
- Feature extraction pipeline

### ✅ Phase 2: Candidate Generation (Retrieval System)
- Two-tower architecture (item + user encoders)
- FAISS index for fast similarity search
- Session-aware user embeddings
- Retrieval evaluation (Recall@K, Hit Rate@K, MRR)

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Setup
1. Download MIND dataset (small or large)
2. Place `behaviors.tsv` and `news.tsv` in `data/raw/`

### Run Phase 1 Pipeline
```bash
python pipeline.py
```

This will:
- Parse MIND impressions into interaction events
- Build user sessions with 30-minute gap threshold
- Simulate dwell time and fatigue signals
- Extract session state features
- Generate training samples with negative sampling

### Run Phase 2 Pipeline
```bash
python phase2_pipeline.py
```

This will:
- Train two-tower model (item + user encoders)
- Precompute item embeddings
- Build FAISS index for fast retrieval
- Evaluate retrieval performance (Recall@K, Hit Rate@K)
- Save retrieval system for Phase 3

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
├── pipeline.py                 # Phase 1 pipeline
├── phase2_pipeline.py          # Phase 2 pipeline
├── requirements.txt
├── phase0_design_doc.md       # Formal problem definition
├── PHASE1_GUIDE.md            # Phase 1 implementation guide
├── PHASE2_GUIDE.md            # Phase 2 implementation guide
├── src/
│   ├── data_loader.py         # MIND dataset loading
│   ├── session_builder.py     # Session construction
│   ├── signal_simulator.py    # Behavioral signal simulation
│   ├── feature_extractor.py   # Session state features
│   ├── negative_sampler.py    # Negative sampling
│   ├── item_encoder.py        # Article encoder (TF-IDF + categories)
│   ├── user_encoder.py        # Session-aware user encoder
│   ├── two_tower_model.py     # Two-tower orchestration
│   ├── faiss_index.py         # FAISS index for retrieval
│   ├── retrieval_system.py    # End-to-end retrieval API
│   └── retrieval_evaluator.py # Retrieval evaluation metrics
├── data/
│   ├── raw/                   # MIND dataset files
│   └── processed/             # Generated sessions, features, models
└── notebooks/                 # Analysis notebooks (coming soon)
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

- [x] Phase 0: Problem Framing
- [x] Phase 1: Dataset & Session Simulation
- [x] Phase 2: Candidate Generation (Retrieval System)
- [ ] Phase 3: Base Ranking Models
- [ ] Phase 4: Multi-Objective Scoring
- [ ] Phase 5: Rule-Based Weight Adaptation
- [ ] Phase 6: Contextual Bandit Integration
- [ ] Phase 7: Evaluation & Visualization
- [ ] Phase 8: Serving Infrastructure

## Dataset

Using [Microsoft MIND](https://msnews.github.io/) (Microsoft News Dataset):
- 1M+ user interactions
- 160K+ news articles
- Rich metadata (categories, entities, abstracts)

## License

MIT

## Contact

For questions or collaboration, please open an issue.
