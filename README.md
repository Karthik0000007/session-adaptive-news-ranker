# Session-Adaptive Multi-Objective News Feed Optimizer

A production-grade personalized news feed ranking system that enforces dynamic trade-offs between engagement, retention, diversity, and novelty based on real-time session context using contextual bandits and counterfactual evaluation.

The core contribution is a **three-layer adaptive decision controller** combining:
1. **Contextual Bandit (LinUCB)** — a lightweight RL policy that dynamically selects weight combinations based on session state to optimize multi-objective rewards.
2. **Conformal Prediction Safety Layer** — distribution-free statistical guarantees for ranking quality with online adaptation to distribution shift.
3. **Rule-Based Fallback** — a deterministic emergency controller ensuring graceful degradation when learned layers fail.

> See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical documentation.

---

## Key Features

- **Two-Tower Retrieval + FAISS** — fast candidate generation (100K+ articles, <50ms latency) with session-aware user embeddings
- **LightGBM Multi-Objective Ranking** — calibrated CTR, dwell time, and retention predictions with comprehensive feature engineering
- **Adaptive Weight Selection** — LinUCB contextual bandit learns optimal trade-offs between engagement, retention, diversity, and novelty
- **Counterfactual Evaluation (IPS/SNIPS)** — off-policy evaluation without live deployment, enabling safe policy validation
- **Greedy Diversity-Aware Reranking** — list-level optimization preventing filter bubbles and improving long-term retention
- **Streamlit Dashboard** — real-time visualization of trade-offs, strategy comparison, and learning dynamics
- **Parquet Telemetry** — columnar per-session logging with drift detection (KS-test + CUSUM)

---

## Architecture Overview

```
User Session
    ↓
Retrieval (Two-Tower + FAISS) → Top-100 candidates (~50ms)
    ↓
Ranking (LightGBM) → CTR, Dwell, Retention scores (~50ms)
    ↓
Decision Layer → Multi-objective scoring (E·R·D·N)
    ↓
Contextual Bandit (LinUCB) → Optimal weight selection
    ↓
Greedy Reranking → Diversity-aware final list
    ↓
Top-20 Recommendations

Parallel (CPU):
    State Extractor (s ∈ ℝ¹¹)
    → Bandit Actor (<0.01ms)
    → Conformal Override (<0.02ms)
    → Rule-Based Fallback
    → Config update (weights / thresholds)
```

---

## Requirements

| Component | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.x |
| LightGBM | 4.0+ |
| FAISS | 1.7+ |
| FastAPI | 0.100+ |
| Redis | 7.x |
| Streamlit | 1.25+ |
| CUDA Toolkit | 11.8+ (optional, CPU works) |
| GPU | Optional (CPU-only supported) |
| OS | Ubuntu 20.04+ or Windows 10/11 |

---

## Installation

**Option 1: Direct**

```bash
git clone <repository-url>
cd session-adaptive-news-ranker

python -m venv venv
source venv/bin/activate        # Linux/macOS
# .\venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

**Option 2: Docker**

```bash
docker build -t news-ranker .
docker-compose up -d
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## Quick Start (End-to-End)

```bash
# 1. Download MIND dataset
#    Place behaviors.tsv and news.tsv in data/raw/

# 2. Process data and build retrieval index
python pipeline.py

# 3. Train ranking models
python train_retrieval.py
python train_ranker.py

# 4. Evaluate decision layer
python evaluate_decision_layer.py

# 5. Train contextual bandit
python train_bandit.py

# 6. Run A/B test simulation
python simulate_ab_test.py

# 7. Start production server
redis-server &
python serve.py

# 8. Open dashboard
streamlit run dashboard.py
# → http://localhost:8501
```

---

## Repository Structure

```
session-adaptive-news-ranker/
├── README.md                    # This file
├── ARCHITECTURE.md              # Technical documentation
├── requirements.txt             # Python dependencies
├── Dockerfile
├── docker-compose.yml
├── config.yaml                  # Configuration
│
├── src/
│   ├── data_loader.py           # MIND dataset loading
│   ├── session_builder.py       # Session construction
│   ├── signal_simulator.py      # Behavioral signal simulation
│   ├── feature_extractor.py     # Session state features
│   ├── negative_sampler.py      # Negative sampling
│   ├── item_encoder.py          # Article encoder (TF-IDF)
│   ├── user_encoder.py          # Session-aware user encoder
│   ├── two_tower_model.py       # Two-tower orchestration
│   ├── faiss_index.py           # FAISS index management
│   ├── retrieval_system.py      # End-to-end retrieval API
│   ├── retrieval_evaluator.py   # Retrieval metrics
│   ├── base_ranker.py           # LightGBM ranking model
│   ├── ranking_features.py      # Feature engineering
│   ├── ranking_system.py        # Multi-objective ranking
│   ├── objective_scorer.py      # Multi-objective scoring
│   ├── weight_adapter.py        # Rule-based weight adaptation
│   ├── decision_layer.py        # Decision layer integration
│   ├── contextual_bandit.py     # LinUCB implementation
│   ├── counterfactual_evaluator.py  # IPS/SNIPS evaluation
│   ├── session_manager.py       # Redis session management
│   └── __init__.py
│
├── scripts/
│   ├── collect_traces.py        # Trace collection
│   ├── train_detection.py       # Detection model training
│   ├── train_segmentation.py    # Segmentation training
│   ├── export_onnx.py           # ONNX export
│   ├── build_tensorrt.py        # TensorRT compilation
│   ├── train_ppo.py             # PPO training
│   ├── train_lyapunov.py        # Lyapunov training
│   ├── calibrate_cp.py          # Conformal calibration
│   ├── eval_agent.py            # Agent evaluation
│   ├── eval_baselines.py        # Baseline comparison
│   ├── stress_test.py           # Stress testing
│   ├── ablation.py              # Ablation studies
│   └── plot_results.py          # Result visualization
│
├── app/
│   └── dashboard.py             # Streamlit dashboard
│
├── models/
│   ├── detection/               # Detection engines
│   ├── segmentation/            # Segmentation engines
│   └── checkpoints/             # RL weights + conformal state
│
├── data/
│   ├── raw/                     # MIND dataset
│   └── processed/               # Generated sessions, features, models
│
├── traces/                      # Logged telemetry (Parquet)
├── results/                     # Figures, tables, experiment logs
├── tests/                       # Unit + integration tests
└── demo/                        # Demo scripts + recordings
```

---

## Evaluation

Run the full evaluation suite:

```bash
# Evaluate all baselines
python scripts/eval_baselines.py --traces traces/ --n-frames 10000

# Evaluate learned agent
python scripts/eval_agent.py --traces traces/ --agent checkpoints/ppo_lyapunov/ --n-frames 10000

# Stress testing
python scripts/stress_test.py --config config.yaml

# Generate results
python scripts/plot_results.py --results results/
```

**Tracked Metrics:**
- P50/P95/P99 latency
- Violation rate (target ≤ 1%)
- Detection quality (mAP@0.5 ≥ 80% of baseline)
- Conformal coverage (≥ 99%)
- Controller overhead (<0.15ms)

**Baselines:**
- Fixed-High-Quality
- Fixed-Low-Latency
- Rule-Based
- PID Controller
- PPO (unconstrained)
- PPO + Lagrangian
- PPO + Lyapunov
- PPO + Lyapunov + Conformal (full system)

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PIPELINE_CONFIG` | `config.yaml` | Pipeline configuration path |
| `CONTROLLER_CONFIG` | `config/controller.yaml` | Controller hyperparameter path |
| `CAMERA_SOURCE` | `0` | Camera device index or video file path |
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |

---

## API Endpoints

```bash
# Rank articles
POST /rank
{
  "user_id": "U12345",
  "k": 20,
  "strategy": "bandit"
}

# Health check
GET /health

# System metrics
GET /metrics

# Log interaction
POST /log
```

---

## Performance Benchmarks

Expected metrics on MIND-small dataset:

| Metric | Value |
|---|---|
| Retrieval Recall@100 | >85% |
| CTR AUC | >0.65 |
| Dwell RMSE | <30s |
| API Latency (P99) | <150ms |
| Bandit Reward | >0.6 |
| Session Length Improvement | +12% vs engagement-only |

---

## Dataset

Using [Microsoft MIND](https://msnews.github.io/) (Microsoft News Dataset):
- 1M+ user interactions
- 160K+ news articles
- Rich metadata (categories, entities, abstracts)

---

## References

- Li et al. *A Contextual-Bandit Approach to Personalized News Article Recommendation.* WWW 2010.
- Dudík et al. *Doubly Robust Policy Evaluation and Learning.* ICML 2011.
- Swaminathan & Joachims. *Counterfactual Risk Minimization.* ICML 2015.
- Gibbs & Candès. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS 2021.

---

## License

MIT

---

## Contact

For questions or collaboration, please open an issue.
