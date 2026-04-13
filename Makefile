.PHONY: install test lint serve dashboard pipeline docker clean

# Install dependencies
install:
	pip install -r requirements.txt

# Run unit tests
test:
	python -m pytest tests/ -v --tb=short

# Lint check
lint:
	python -m flake8 src/ --max-line-length=120 --ignore=E501,W503

# Run API server
serve:
	uvicorn app.serve:app --host 0.0.0.0 --port 8000 --reload

# Run Streamlit dashboard
dashboard:
	streamlit run app/dashboard.py --server.port=8501 --server.address=0.0.0.0

# Run data pipeline (Phase 1)
pipeline:
	python scripts/pipeline.py

# Train all models
train: train-retrieval train-ranker train-bandit

train-retrieval:
	python scripts/train_retrieval.py

train-ranker:
	python scripts/train_ranker.py

train-bandit:
	python scripts/train_bandit.py

# Evaluate
evaluate:
	python scripts/evaluate_decision_layer.py

# A/B test simulation
ab-test:
	python scripts/simulate_ab_test.py

# Docker
docker-build:
	docker build -t news-ranker .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Clean generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache
