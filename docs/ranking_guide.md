# Phase 3 Implementation Guide: Base Ranking Models

## Overview

Phase 3 builds ranking models that predict high-quality signals (CTR, dwell time, retention) for ~100 candidates retrieved in Phase 2. These models produce calibrated scores that feed into the multi-objective decision layer in Phase 4.

**Key Principle:** Ranking models produce signals, not final decisions. The decision layer (Phase 4+) combines these signals optimally.

## Architecture

```
Candidates (from Phase 2)
    ↓
Feature Engineering
    ↓
┌─────────────────────────────────┐
│  CTR Model      (LightGBM)      │ → ctr_score
│  Dwell Model    (LightGBM)      │ → dwell_score
│  Retention Model (LightGBM)     │ → retention_score
└─────────────────────────────────┘
    ↓
Calibrated Signals → Phase 4
```

## Components Built

### 1. Base Ranker (`src/base_ranker.py`)

**Purpose:** LightGBM-based ranking model with calibration

**Features:**
- Handles both binary (CTR, retention) and regression (dwell) tasks
- Automatic categorical feature encoding
- Platt scaling for probability calibration
- Feature importance extraction

**Why LightGBM?**
- Fast training and inference
- Handles mixed feature types (numerical + categorical)
- Strong baseline performance
- Production-proven
- Built-in early stopping

**Calibration:**
- Uses Platt scaling (sigmoid calibration)
- Makes probabilities comparable across objectives
- Critical for multi-objective optimization

### 2. Ranking Feature Builder (`src/ranking_features.py`)

**Purpose:** Transform raw data into model-ready features

**Feature Categories:**

#### User Features
- `user_total_clicks`: Historical click count
- `user_avg_dwell`: Average dwell time across sessions
- `user_num_sessions`: Number of sessions

#### Item Features
- `item_category`: Article category
- `item_subcategory`: Article subcategory
- `item_popularity`: Global click count
- `item_ctr`: Historical CTR
- `item_impressions`: Total impressions

#### Session Features (Critical)
- `session_length`: Number of interactions so far
- `session_avg_dwell`: Average dwell in current session
- `session_click_rate`: Click rate in current session
- `session_skip_rate`: Skip rate in current session
- `session_click_entropy`: Category diversity
- `session_fatigue`: Fatigue indicator
- `session_time_of_day`: Hour (0-23)
- `session_day_of_week`: Day (0-6)

#### Interaction Features (Highest Signal)
- `user_category_affinity`: User's clicks in this category
- `has_clicked_before`: Binary flag
- `similarity_to_recent`: Similarity to last clicked item
- `num_recent_clicks`: Recent interaction count

**Why Interaction Features Matter:**
- Capture user-item fit
- Often highest feature importance
- Enable personalization

### 3. Ranking System (`src/ranking_system.py`)

**Purpose:** Orchestrate multiple ranking models

**Three Models:**

1. **CTR Model** (Binary Classification)
   - Predicts: P(click | user, item, session)
   - Metric: AUC, LogLoss
   - Calibrated: Yes

2. **Dwell Model** (Regression)
   - Predicts: Expected dwell time (log-transformed)
   - Metric: RMSE, MAE
   - Trained on: Clicked items only
   - Calibrated: Normalized to [0, 1]

3. **Retention Model** (Binary Classification)
   - Predicts: P(session continues | user, item, session)
   - Metric: AUC, LogLoss
   - Simplified: Uses click as proxy (upgrade in production)
   - Calibrated: Yes

**Unified Prediction Interface:**
```python
scores = ranking_system.predict(
    user_id='U123',
    article_id='N456',
    session_state={...},
    article_metadata={...},
    clicked_history=['N1', 'N2']
)

# Returns:
# {
#     'ctr_score': 0.65,
#     'dwell_score': 0.42,
#     'retention_score': 0.58
# }
```

## How to Run

### 1. Ensure Phase 1 Complete

```bash
# Verify Phase 1 outputs exist
ls data/processed/
# Should show: sessions.pkl, training_samples.pkl
```

### 2. Run Phase 3 Pipeline

```bash
python phase3_pipeline.py
```

**Expected Output:**
```
Phase 3: Base Ranking Models
============================================================
[1/5] Loading MIND dataset...
  - Loaded X articles
[2/5] Loading Phase 1 data...
  - Loaded Y users
  - Loaded Z training samples
[3/5] Building feature engineering pipeline...
  Building user profiles...
    - Built profiles for Y users
  Building item statistics...
    - Built stats for X items
[4/5] Preparing training data...
  - Prepared Z training samples
  - Features: N

Training Data Summary:
  - Total samples: Z
  - Positive clicks: P
  - Click rate: 0.XXXX
  - Avg dwell (clicked): XX.XXs

[5/5] Training ranking models...

============================================================
Training CTR Model
============================================================
Training with N features
Train size: T, Val size: V
[LightGBM training logs...]

CTR Model Validation Metrics:
  AUC: 0.XXXX
  LogLoss: 0.XXXX

============================================================
Training Dwell Model
============================================================
[LightGBM training logs...]

Dwell Model Validation Metrics:
  RMSE: 0.XXXX
  MAE: 0.XXXX

============================================================
Training Retention Model
============================================================
[LightGBM training logs...]

Retention Model Validation Metrics:
  AUC: 0.XXXX
  LogLoss: 0.XXXX

============================================================
Top 10 Features by Importance
============================================================

CTR Model:
                    feature  importance
  user_category_affinity      1234.56
       item_popularity         987.65
      session_fatigue          876.54
                  ...             ...

[Similar for Dwell and Retention models]

Ranking system saved to: data/processed/ranking_models
```

### 3. Verify Outputs

```bash
ls data/processed/ranking_models/
# Should show:
# - ctr_model_lgb.txt
# - ctr_model_meta.pkl
# - dwell_model_lgb.txt
# - dwell_model_meta.pkl
# - retention_model_lgb.txt
# - retention_model_meta.pkl
# - feature_builder.pkl
# - training_summary.json
```

## Configuration

Edit `config.yaml` to customize:

```yaml
ranking:
  model_type: 'lightgbm'
  calibration: true
```

## Output Format

### Prediction Output
```python
{
    'ctr_score': 0.65,        # P(click) ∈ [0, 1]
    'dwell_score': 0.42,      # Normalized dwell ∈ [0, 1]
    'retention_score': 0.58   # P(session continues) ∈ [0, 1]
}
```

### Saved Models
```
ranking_models/
├── ctr_model_lgb.txt           # LightGBM model (text format)
├── ctr_model_meta.pkl          # Metadata (encoders, calibration)
├── dwell_model_lgb.txt
├── dwell_model_meta.pkl
├── retention_model_lgb.txt
├── retention_model_meta.pkl
├── feature_builder.pkl         # Feature engineering pipeline
└── training_summary.json       # Training statistics
```

## Key Design Decisions

### Why Separate Models?
- Cleaner signals: Each model optimizes its own objective
- Easier debugging: Can inspect each model independently
- Better interpretability: Feature importance per objective
- Flexible: Can upgrade models independently

**Alternative:** Multi-task learning (single model, multiple outputs)
- Pros: Shared representations, faster inference
- Cons: Harder to debug, objectives may conflict
- Recommendation: Start with separate models, upgrade later

### Why Time-Based Split?
- Simulates production: Train on past, predict future
- Prevents leakage: No future information in training
- Realistic evaluation: Matches deployment scenario

**Bad:** Random split (leaks future information)
**Good:** Time-based split (80% train, 20% validation)

### Why Log-Transform Dwell Time?
- Dwell time is right-skewed (most short, few long)
- Log transform normalizes distribution
- Improves model performance
- Remember to inverse transform predictions: `np.expm1(pred)`

### Why Train Dwell on Clicked Items Only?
- Non-clicked items have minimal dwell (scan time)
- Clicked items have meaningful dwell (reading time)
- Separate models: CTR predicts click, Dwell predicts engagement given click

### Why Calibration?
- Raw model scores may not be well-calibrated probabilities
- Calibration ensures: predicted 0.7 → 70% actual click rate
- Critical for multi-objective optimization (comparing apples to apples)
- Platt scaling: Fast, effective for tree models

## Evaluation Interpretation

### Good Metrics

**CTR Model:**
- AUC > 0.70: Good discrimination
- LogLoss < 0.5: Well-calibrated

**Dwell Model:**
- RMSE < 1.0: Reasonable error (log scale)
- MAE < 0.7: Good average error

**Retention Model:**
- AUC > 0.65: Decent prediction
- LogLoss < 0.6: Acceptable calibration

### What to Do If Metrics Are Low

**Low AUC:**
- Add more interaction features
- Check for data leakage
- Try feature engineering (ratios, differences)

**High LogLoss:**
- Model not calibrated → check calibration step
- Overfitting → reduce model complexity

**High RMSE/MAE:**
- Check label distribution (outliers?)
- Try different transformations
- Add more item features

## Feature Importance Insights

**Typical Top Features:**
1. `user_category_affinity`: User's preference for category
2. `item_popularity`: Popular items get more clicks
3. `session_fatigue`: Fatigued users behave differently
4. `similarity_to_recent`: Recency matters
5. `item_ctr`: Historical performance

**If Unexpected:**
- Session features dominate → Good! Session-awareness working
- Only item features → User features weak, need better profiling
- Random features high → Possible overfitting or data issues

## Next Steps

Phase 3 outputs feed into:
- **Phase 4**: Multi-objective scoring (combine signals)
- **Phase 5**: Rule-based weight adaptation (session-aware)
- **Phase 6**: Contextual bandit (learned weights)

## Deliverables Checklist

- [x] CTR model (LightGBM + calibration)
- [x] Dwell model (LightGBM + normalization)
- [x] Retention model (LightGBM + calibration)
- [x] Feature engineering pipeline
- [x] User profiles and item statistics
- [x] Time-based train/val split
- [x] Model evaluation metrics
- [x] Feature importance analysis
- [x] Unified prediction interface
- [x] Model serialization

## Architecture Alignment

This phase implements:
- **Base Ranking:** Produces engagement, retention, diversity signals
- **Calibrated Scores:** Enables fair multi-objective comparison
- **Session-Aware Features:** Captures user state dynamics
- **Production-Ready:** Fast inference, serializable models
- **Foundation for Decision Layer:** Clean signals for Phase 4+
