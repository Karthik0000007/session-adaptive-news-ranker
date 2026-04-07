# Git Commands for Session-Adaptive News Ranker

This document provides git commands for committing each phase of the project.

## Phase 1: Data Processing Pipeline

```bash
git add pipeline.py
git add src/data_loader.py src/session_builder.py src/signal_simulator.py
git add src/feature_extractor.py src/negative_sampler.py
git commit -m "Add data processing pipeline with session construction and signal simulation"

git add docs/data_pipeline_guide.md
git commit -m "Add data pipeline documentation"
```

## Phase 2: Retrieval System

```bash
git add train_retrieval.py
git add src/item_encoder.py src/user_encoder.py src/two_tower_model.py
git add src/faiss_index.py src/retrieval_system.py src/retrieval_evaluator.py
git commit -m "Add two-tower retrieval system with FAISS indexing"

git add docs/retrieval_guide.md
git commit -m "Add retrieval system documentation"
```

## Phase 3: Base Ranking Models

```bash
git add train_ranker.py
git add src/base_ranker.py src/ranking_features.py src/ranking_system.py
git commit -m "Add LightGBM ranking models with calibration"

git add docs/ranking_guide.md
git commit -m "Add ranking models documentation"
```

## Phase 4 & 5: Multi-Objective Decision Layer

```bash
git add src/objective_scorer.py src/weight_adapter.py src/decision_layer.py
git commit -m "Add multi-objective scoring and rule-based weight adaptation"

git add evaluate_decision_layer.py
git commit -m "Add decision layer evaluation pipeline"
```

## Phase 6 & 7: Contextual Bandit & Counterfactual Evaluation

```bash
git add src/contextual_bandit.py src/counterfactual_evaluator.py
git commit -m "Add LinUCB contextual bandit with IPS/SNIPS evaluation"

git add train_bandit.py
git commit -m "Add bandit training pipeline"
```

## Phase 8 & 9: Simulation & Serving

```bash
git add simulate_ab_test.py
git commit -m "Add A/B testing simulation framework with user behavior modeling"

git add serve.py src/session_manager.py
git commit -m "Add FastAPI production serving with Redis session management"

git add Dockerfile docker-compose.yml
git commit -m "Add Docker configuration for containerized deployment"
```

## Phase 10: Visualization Dashboard

```bash
git add dashboard.py
git commit -m "Add Streamlit visualization dashboard with trade-off analysis"
```

## Phase 11: Production Readiness

```bash
git add docs/production_guide.md
git commit -m "Add production readiness guide with failure modes and monitoring"
```

## Documentation & Configuration

```bash
git add DESIGN.md ARCHITECTURE.md
git commit -m "Add comprehensive design and architecture documentation"

git add README.md
git commit -m "Update README with complete project overview"

git add requirements.txt
git commit -m "Update dependencies for all phases"

git add config.yaml
git commit -m "Add project configuration"
```

## Complete Project Commit

If you want to commit everything at once:

```bash
git add .
git commit -m "Complete session-adaptive news ranker implementation

- Data processing pipeline with session construction
- Two-tower retrieval system with FAISS
- LightGBM ranking models (CTR, dwell, retention)
- Multi-objective decision layer with adaptive weighting
- LinUCB contextual bandit with counterfactual evaluation
- A/B testing simulation framework
- Production FastAPI serving with Redis
- Interactive Streamlit dashboard
- Comprehensive documentation and Docker deployment"
```

## Tagging Releases

```bash
# Tag major milestones
git tag -a v0.1.0 -m "Phase 1-3: Core ML pipeline"
git tag -a v0.2.0 -m "Phase 4-5: Decision layer"
git tag -a v0.3.0 -m "Phase 6-7: Bandit and evaluation"
git tag -a v1.0.0 -m "Phase 8-11: Production-ready system"

# Push tags
git push origin --tags
```

## Branch Strategy (Optional)

If using feature branches:

```bash
# Create feature branches
git checkout -b feature/retrieval
git checkout -b feature/ranking
git checkout -b feature/decision-layer
git checkout -b feature/bandit
git checkout -b feature/serving
git checkout -b feature/visualization

# Merge to main
git checkout main
git merge feature/retrieval
git merge feature/ranking
# ... etc
```

## Viewing History

```bash
# View commit history
git log --oneline --graph --all

# View changes in a specific phase
git log --oneline --grep="bandit"

# View file history
git log --follow -- src/contextual_bandit.py
```
