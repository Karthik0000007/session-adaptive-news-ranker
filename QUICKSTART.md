# Quick Start Guide

Complete walkthrough for running the Session-Adaptive News Ranker from scratch.

## Prerequisites

### System Requirements
- Python 3.10+
- 8GB RAM minimum (16GB recommended)
- 10GB disk space
- Redis (for serving)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd session-adaptive-news-ranker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

### Download MIND Dataset

1. Visit [Microsoft MIND](https://msnews.github.io/)
2. Download MIND-small or MIND-large
3. Extract files to `data/raw/`:
   - `behaviors.tsv`
   - `news.tsv`

```bash
# Create directories
mkdir -p data/raw data/processed
```

## Complete Pipeline Execution

### Step 1: Data Processing (Phase 1)

```bash
python pipeline.py
```

**Expected Output:**
- `data/processed/sessions.pkl`
- `data/processed/training_samples.pkl`
- `data/processed/dataset_stats.json`

**Runtime:** ~10-15 minutes

**What it does:**
- Parses MIND impressions
- Constructs user sessions (30-min gap)
- Simulates dwell time and fatigue
- Extracts session features

---

### Step 2: Train Retrieval System (Phase 2)

```bash
python train_retrieval.py
```

**Expected Output:**
- `data/processed/retrieval_model/`
  - `model/item_encoder.pkl`
  - `model/user_encoder.pkl`
  - `index/faiss.index`
  - `eval_results.json`

**Runtime:** ~20-30 minutes

**What it does:**
- Trains two-tower model
- Builds FAISS index
- Evaluates Recall@K

---

### Step 3: Train Ranking Models (Phase 3)

```bash
python train_ranker.py
```

**Expected Output:**
- `data/processed/ranking_models/`
  - `ctr_model_lgb.txt`
  - `dwell_model_lgb.txt`
  - `retention_model_lgb.txt`
  - `training_summary.json`

**Runtime:** ~30-45 minutes

**What it does:**
- Trains CTR, dwell, retention models
- Calibrates predictions
- Evaluates AUC, RMSE

---

### Step 4: Evaluate Decision Layer (Phase 4-5)

```bash
python evaluate_decision_layer.py
```

**Expected Output:**
- `data/processed/decision_layer/evaluation_results.json`

**Runtime:** ~15-20 minutes

**What it does:**
- Tests multi-objective scoring
- Compares fixed vs rule-based weights
- Measures all objectives (E, R, D, N)

---

### Step 5: Train Contextual Bandit (Phase 6-7)

```bash
python train_bandit.py
```

**Expected Output:**
- `data/processed/bandit_model/`
  - `bandit_state.pkl`
  - `training_logs.pkl`
  - `evaluation_results.json`

**Runtime:** ~20-30 minutes

**What it does:**
- Trains LinUCB bandit
- Learns weight selection policy
- Evaluates with IPS/SNIPS

---

### Step 6: Run A/B Test Simulation (Phase 8)

```bash
python simulate_ab_test.py
```

**Expected Output:**
- `data/processed/ab_test_results/results.json`
- Console output with comparison table

**Runtime:** ~30-45 minutes

**What it does:**
- Simulates 1000+ user sessions
- Compares all strategies
- Generates metrics

---

### Step 7: Start Production Server (Phase 9)

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start API
python serve.py
```

**API Available at:** `http://localhost:8000`

**Test the API:**
```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "U12345",
    "k": 20,
    "strategy": "bandit"
  }'
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

---

### Step 8: Launch Dashboard (Phase 10)

```bash
streamlit run dashboard.py
```

**Dashboard Available at:** `http://localhost:8501`

**Features:**
- Trade-off curves
- Session evolution
- Strategy comparison
- Bandit learning curves

---

## Docker Deployment (Alternative)

### Using Docker Compose

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`
- Redis: `localhost:6379`

---

## Verification Checklist

After running the pipeline, verify:

- [ ] Sessions created: `ls data/processed/sessions.pkl`
- [ ] FAISS index built: `ls data/processed/retrieval_model/index/`
- [ ] Ranking models trained: `ls data/processed/ranking_models/*.txt`
- [ ] Bandit trained: `ls data/processed/bandit_model/bandit_state.pkl`
- [ ] A/B results generated: `ls data/processed/ab_test_results/results.json`
- [ ] API responds: `curl http://localhost:8000/health`
- [ ] Dashboard loads: Open `http://localhost:8501`

---

## Troubleshooting

### Issue: Out of Memory

**Solution:**
- Use MIND-small instead of MIND-large
- Reduce batch size in `config.yaml`
- Close other applications

### Issue: Redis Connection Error

**Solution:**
```bash
# Check if Redis is running
redis-cli ping

# Start Redis
redis-server

# Or use Docker
docker run -d -p 6379:6379 redis:7-alpine
```

### Issue: FAISS Import Error

**Solution:**
```bash
# Reinstall FAISS
pip uninstall faiss-cpu
pip install faiss-cpu
```

### Issue: Model Not Found

**Solution:**
- Ensure previous steps completed successfully
- Check `data/processed/` directory structure
- Re-run training scripts

---

## Configuration

Edit `config.yaml` to customize:

```yaml
data:
  session_gap_minutes: 30
  max_session_length: 50
  negative_samples: 4

retrieval:
  embedding_dim: 128
  batch_size: 256

ranking:
  model_type: lightgbm
  calibration: true

bandit:
  alpha: 0.5  # Exploration parameter
  num_iterations: 10000
```

---

## Next Steps

1. **Analyze Results:**
   - Review A/B test comparison
   - Examine trade-off curves
   - Study bandit learning dynamics

2. **Experiment:**
   - Adjust weights in `config.yaml`
   - Try different bandit parameters
   - Test new strategies

3. **Deploy:**
   - Use Docker for production
   - Set up monitoring
   - Configure alerts

4. **Extend:**
   - Add new objectives
   - Implement deep learning models
   - Integrate real user feedback

---

## Performance Benchmarks

Expected metrics on MIND-small:

| Metric | Value |
|---|---|
| Retrieval Recall@100 | >85% |
| CTR AUC | >0.65 |
| Dwell RMSE | <30s |
| API Latency (P99) | <150ms |
| Bandit Reward | >0.6 |

---

## Support

For issues or questions:
1. Check `docs/` directory for detailed guides
2. Review `ARCHITECTURE.md` for system design
3. See `docs/production_guide.md` for operational issues
4. Open an issue on GitHub

---

**Last Updated:** 2026-04-07  
**Version:** 1.0
