# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-04-13

### Fixed — P0 Critical
- **Retention label** (Fix 1): Replaced broken `label_retention = label_click` with actual session-continuation labels computed from timestamp gaps. The 4-objective system now has 4 genuinely independent objectives.
- **Eval metrics** (Fix 2): Removed copy-paste DTA metrics (mse, rmse, ci, pearson, spearman) and replaced with proper ranking metrics (auc, logloss, ndcg@10, recall@10, mrr).
- **Position bias** (Fix 3): Added position features (display_position, log_discount, is_top3, is_top10) and IPS sample weighting to correct for position-induced click bias.
- **Import mismatch** (Fix 4): Fixed `serve.py` importing non-existent `ContextualBandit` — corrected to `LinUCB`.

### Fixed — P1 Medium
- **Redis security** (Fix 5): Replaced `pickle` serialization with JSON in `serve.py` and `session_manager.py` to eliminate deserialization vulnerability.
- **Cache determinism** (Fix 6): Replaced `hash()` (non-deterministic across restarts) with `hashlib.md5` for cache keys.
- **Cold-start** (Fix 11): Replaced fake `popular_0..popular_k` articles with pre-computed diverse set from actual article corpus.

### Added — P1 Medium
- **Unit tests** (Fix 7): 30+ tests covering objective scorer, weight adapter, LinUCB bandit, drift detector, Japanese tokenizer, counterfactual evaluator, and position bias features.
- **Drift detector** (Fix 8): `src/drift_detector.py` with KS-test, CUSUM, and multi-feature monitoring for production distribution shift detection.
- **Japanese tokenizer** (Fix 9): `src/japanese_tokenizer.py` with SudachiPy backend and regex fallback for Japanese text processing.
- **APPI compliance** (Fix 10): `DELETE /user/{user_id}` for data deletion and `GET /privacy/purpose` for usage purpose notification (APPI Articles 18, 30).

### Added — P2 Nice-to-have
- **CI/CD pipeline** (Fix 12): `.github/workflows/ci.yml` with test, lint, and Docker build verification.
- **Prometheus monitoring** (Fix 13): Custom metrics (ranking latency histogram, cache hit/miss counters, active sessions gauge) and FastAPI instrumentator.
- **Dockerfile upgrade** (Fix 14): Multi-stage build with non-root user for production security.

### Added — Config & Tooling
- `config/serving.yaml`: Serving config with latency budgets and fallback chain
- `config/monitoring.yaml`: Alert thresholds for business, system, drift, and bandit metrics
- `config/pipeline.yaml`: Data freshness strategy and retention label config
- `Makefile`: Common development tasks (install, test, lint, serve, train, docker)
- `CHANGELOG.md`: This file

## [1.0.0] - 2026-04-07

### Initial Release
- End-to-end news ranking pipeline (data → training → serving)
- Two-tower retrieval with FAISS
- LightGBM ranking models (CTR, dwell, retention)
- Multi-objective scoring with rule-based and contextual bandit weighting
- FastAPI serving with Redis session management
- Streamlit dashboard
- Docker deployment
- A/B testing simulation
- Counterfactual evaluation (IPS/SNIPS)
