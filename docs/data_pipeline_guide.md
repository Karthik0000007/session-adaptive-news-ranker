# Phase 1 Implementation Guide

## Overview

Phase 1 transforms the MIND dataset from impression-based format into realistic session-based training data with simulated behavioral signals.

## What Was Built

### Core Components

1. **Data Loader** (`src/data_loader.py`)
   - Parses MIND `behaviors.tsv` and `news.tsv`
   - Converts impression format to individual interaction events
   - Extracts article metadata

2. **Session Builder** (`src/session_builder.py`)
   - Groups events into sessions using 30-minute gap threshold
   - Filters sessions by minimum length
   - Computes session statistics

3. **Signal Simulator** (`src/signal_simulator.py`)
   - Simulates dwell time (log-normal for clicks, uniform for non-clicks)
   - Computes fatigue scores based on dwell and skip patterns
   - Adds realistic behavioral signals missing from MIND

4. **Feature Extractor** (`src/feature_extractor.py`)
   - Extracts session state at each interaction point
   - Computes click entropy, recent categories, temporal features
   - Creates training samples with full feature vectors

5. **Negative Sampler** (`src/negative_sampler.py`)
   - Generates negative samples for ranking training
   - Excludes already-seen articles
   - Configurable negative ratio

### Pipeline Flow

```
MIND Dataset
    ↓
Parse Impressions → Individual Events
    ↓
Build Sessions → Time-based Grouping
    ↓
Simulate Signals → Dwell Time + Fatigue
    ↓
Extract Features → Session State Vectors
    ↓
Generate Samples → Positive + Negatives
    ↓
Save Outputs → sessions.pkl + training_samples.pkl
```

## How to Run

### 1. Setup Environment

```bash
# Linux/Mac
bash setup.sh

# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download MIND Dataset

Visit: https://msnews.github.io/

Download either:
- **MINDsmall** (~50MB, good for testing)
- **MINDlarge** (~500MB, full dataset)

Extract and place these files in `data/raw/`:
- `behaviors.tsv`
- `news.tsv`

### 3. Run Pipeline

```bash
python pipeline.py
```

Expected output:
```
Phase 1: Dataset & Session Simulation
============================================================
[1/7] Initializing components...
[2/7] Loading MIND dataset...
  - Loaded X behavior records
  - Loaded Y news articles
[3/7] Parsing impressions into events...
  - Generated Z interaction events
[4/7] Building sessions...
  - Total users: ...
  - Total sessions: ...
[5/7] Simulating behavioral signals...
[6/7] Extracting features and creating training samples...
[7/7] Saving outputs...
============================================================
Phase 1 Complete!
```

### 4. Verify Outputs

Check `data/processed/`:
- `sessions.pkl` - Sessionized data with simulated signals
- `training_samples.pkl` - Feature vectors ready for training
- `dataset_stats.json` - Summary statistics

## Configuration

Edit `config.yaml` to customize:

```yaml
session:
  gap_minutes: 30  # Session boundary (increase for longer sessions)
  min_session_length: 2  # Filter very short sessions

simulation:
  dwell_time:
    clicked_mean: 2.0  # Adjust reading time distribution
    clicked_sigma: 0.5
  negative_samples: 10  # More negatives = harder training

features:
  recent_categories_k: 5  # Track last K categories
```

## Output Format

### Session Structure
```python
{
    'user_123': [
        [  # Session 1
            {
                'article_id': 'N12345',
                'clicked': 1,
                'timestamp': datetime(...),
                'dwell_time': 45.2  # Simulated
            },
            ...
        ],
        [...]  # Session 2
    ]
}
```

### Training Sample Structure
```python
{
    'user_id': 'U123',
    'article_id': 'N456',
    
    # Session features
    'session_length': 5,
    'avg_dwell_time': 32.5,
    'click_rate': 0.6,
    'skip_rate': 0.4,
    'click_entropy': 1.2,
    'fatigue_score': 0.3,
    'time_of_day': 14,
    'day_of_week': 2,
    
    # Article features
    'article_category': 'sports',
    'article_subcategory': 'football',
    
    # Labels
    'label_click': 1,
    'label_dwell': 45.2
}
```

## Key Design Decisions

### Why 30-minute session gap?
Industry standard for news consumption. Users typically browse in bursts.

### Why log-normal for dwell time?
Realistic distribution for reading time - most articles read quickly, some read deeply.

### Why simulate dwell time?
MIND only has clicks. Dwell time is critical for engagement modeling and multi-objective optimization.

### Why negative sampling?
Ranking models need to learn relative preferences. Negatives provide contrastive signal.

## Validation Checks

After running, verify:

1. **Session stats look reasonable**
   - Avg session length: 3-10 interactions
   - Sessions per user: 1-5

2. **Dwell time distribution**
   - Clicked: Mean ~60-120 seconds
   - Not clicked: Mean ~1-2 seconds

3. **Training samples**
   - Positive:negative ratio matches config
   - No missing features

## Common Issues

### "File not found: behaviors.tsv"
→ Download MIND dataset and place in `data/raw/`

### "No sessions generated"
→ Check `min_session_length` in config (try setting to 1)

### "Memory error"
→ Use MINDsmall instead of MINDlarge, or process in batches

### "Import errors"
→ Activate virtual environment and reinstall requirements

## Next Steps

Phase 1 outputs feed into:
- **Phase 2**: Feature engineering (embeddings, user profiles)
- **Phase 3**: Base ranking models (LightGBM, neural networks)
- **Phase 4**: Multi-objective scoring implementation

## Deliverables Checklist

- [x] Sessionized dataset with temporal ordering
- [x] Simulated behavioral signals (dwell, fatigue)
- [x] Session state features at each interaction point
- [x] Training samples with positive + negative pairs
- [x] Reproducible pipeline with configuration
- [x] Statistics and validation outputs

## Architecture Alignment

This phase implements the data foundation for:
- Session-aware ranking (temporal context)
- Multi-objective optimization (engagement + retention signals)
- Contextual bandit (session state features)
- Counterfactual evaluation (logged interaction data)
