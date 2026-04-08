"""
Phase 9: Production Serving System

FastAPI-based serving layer with Redis session management.
Provides low-latency ranking endpoint with fallback logic.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import redis
import pickle
import json
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from src.retrieval_system import RetrievalSystem
from src.ranking_system import RankingSystem
from src.decision_layer import DecisionLayer
from src.weight_adapter import WeightAdapter
from src.contextual_bandit import ContextualBandit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Loading models...")
    app_state['retrieval'] = load_retrieval_system()
    app_state['ranking'] = load_ranking_system()
    app_state['decision_layer'] = load_decision_layer()
    app_state['redis_client'] = init_redis()
    logger.info("Models loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if 'redis_client' in app_state:
        app_state['redis_client'].close()


app = FastAPI(
    title="Session-Adaptive News Ranker",
    description="Production serving system for multi-objective news ranking",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response models
class SessionState(BaseModel):
    """Session state features"""
    session_length: int = Field(default=0, ge=0)
    avg_dwell_time: float = Field(default=0.0, ge=0.0)
    click_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    skip_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    click_entropy: float = Field(default=0.0, ge=0.0)
    fatigue: float = Field(default=0.0, ge=0.0, le=1.0)
    time_of_day: int = Field(default=12, ge=0, le=23)
    day_of_week: int = Field(default=3, ge=0, le=6)


class RankRequest(BaseModel):
    """Ranking request"""
    user_id: str
    session_state: Optional[SessionState] = None
    k: int = Field(default=20, ge=1, le=100)
    strategy: str = Field(default='bandit', pattern='^(baseline|rule_based|bandit)$')


class ArticleResponse(BaseModel):
    """Single article in response"""
    article_id: str
    title: str
    category: str
    score: float
    ctr_score: float
    dwell_score: float
    retention_score: float


class RankResponse(BaseModel):
    """Ranking response"""
    articles: List[ArticleResponse]
    weights: List[float]
    strategy: str
    latency_ms: float


# System initialization
def load_retrieval_system() -> RetrievalSystem:
    """Load retrieval system"""
    model_dir = Path('data/processed/retrieval_model')
    if not model_dir.exists():
        raise RuntimeError(f"Retrieval model not found at {model_dir}")
    return RetrievalSystem.load(model_dir)


def load_ranking_system() -> RankingSystem:
    """Load ranking system"""
    model_dir = Path('data/processed/ranking_models')
    if not model_dir.exists():
        raise RuntimeError(f"Ranking models not found at {model_dir}")
    return RankingSystem.load(model_dir)


def load_decision_layer() -> Dict:
    """Load decision layer with multiple strategies"""
    ranking_system = app_state.get('ranking')
    
    strategies = {}
    
    # Baseline: fixed weights
    strategies['baseline'] = DecisionLayer(
        ranking_system=ranking_system,
        weight_adapter=None,
        fixed_weights=[0.4, 0.3, 0.2, 0.1]
    )
    
    # Rule-based: session-adaptive
    strategies['rule_based'] = DecisionLayer(
        ranking_system=ranking_system,
        weight_adapter=WeightAdapter(),
        fixed_weights=None
    )
    
    # Bandit: LinUCB
    bandit_path = Path('data/processed/bandit_model/bandit_state.pkl')
    if bandit_path.exists():
        with open(bandit_path, 'rb') as f:
            bandit = pickle.load(f)
    else:
        logger.warning("Bandit model not found, using default")
        bandit = ContextualBandit(alpha=0.5)
    
    strategies['bandit'] = {
        'decision_layer': DecisionLayer(
            ranking_system=ranking_system,
            weight_adapter=None,
            fixed_weights=None
        ),
        'bandit': bandit
    }
    
    return strategies


def init_redis() -> redis.Redis:
    """Initialize Redis client"""
    try:
        client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=False
        )
        client.ping()
        logger.info("Redis connected")
        return client
    except redis.ConnectionError:
        logger.warning("Redis not available, using in-memory fallback")
        return None


# Session management
def get_session_state(user_id: str, provided_state: Optional[SessionState]) -> Dict:
    """Get session state from Redis or use provided"""
    if provided_state:
        return provided_state.dict()
    
    redis_client = app_state.get('redis_client')
    if redis_client:
        try:
            state_bytes = redis_client.get(f"session:{user_id}")
            if state_bytes:
                return pickle.loads(state_bytes)
        except Exception as e:
            logger.error(f"Redis error: {e}")
    
    # Default state for cold start
    return SessionState().dict()


def update_session_state(user_id: str, state: Dict):
    """Update session state in Redis"""
    redis_client = app_state.get('redis_client')
    if redis_client:
        try:
            redis_client.setex(
                f"session:{user_id}",
                3600,  # 1 hour TTL
                pickle.dumps(state)
            )
        except Exception as e:
            logger.error(f"Redis error: {e}")


# Caching
def get_cache_key(user_id: str, state: Dict) -> str:
    """Generate cache key"""
    state_hash = hash(frozenset(state.items()))
    return f"rank:{user_id}:{state_hash}"


def get_cached_result(cache_key: str) -> Optional[Dict]:
    """Get cached ranking result"""
    redis_client = app_state.get('redis_client')
    if redis_client:
        try:
            result_bytes = redis_client.get(cache_key)
            if result_bytes:
                return pickle.loads(result_bytes)
        except Exception as e:
            logger.error(f"Cache error: {e}")
    return None


def cache_result(cache_key: str, result: Dict, ttl: int = 300):
    """Cache ranking result"""
    redis_client = app_state.get('redis_client')
    if redis_client:
        try:
            redis_client.setex(cache_key, ttl, pickle.dumps(result))
        except Exception as e:
            logger.error(f"Cache error: {e}")


# Main ranking endpoint
@app.post("/rank", response_model=RankResponse)
async def rank_articles(request: RankRequest):
    """
    Rank articles for a user session
    
    Latency target: <150ms (P99)
    """
    start_time = time.time()
    
    try:
        # Get session state
        session_state = get_session_state(request.user_id, request.session_state)
        
        # Check cache
        cache_key = get_cache_key(request.user_id, session_state)
        cached = get_cached_result(cache_key)
        if cached:
            cached['latency_ms'] = (time.time() - start_time) * 1000
            return cached
        
        # Retrieval
        retrieval_start = time.time()
        candidates = app_state['retrieval'].retrieve(
            request.user_id,
            session_state,
            k=100
        )
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        if not candidates:
            # Cold start fallback
            return cold_start_fallback(request, start_time)
        
        # Ranking
        ranking_start = time.time()
        ranked_list, weights = rank_with_strategy(
            request.strategy,
            candidates,
            session_state
        )
        ranking_time = (time.time() - ranking_start) * 1000
        
        # Format response
        articles = [
            ArticleResponse(
                article_id=item['article_id'],
                title=item.get('title', ''),
                category=item.get('category', ''),
                score=item['final_score'],
                ctr_score=item['ctr_score'],
                dwell_score=item['dwell_score'],
                retention_score=item['retention_score']
            )
            for item in ranked_list[:request.k]
        ]
        
        total_time = (time.time() - start_time) * 1000
        
        response = RankResponse(
            articles=articles,
            weights=weights,
            strategy=request.strategy,
            latency_ms=total_time
        )
        
        # Cache result
        cache_result(cache_key, response.dict())
        
        # Log latency
        logger.info(
            f"Ranking completed: retrieval={retrieval_time:.1f}ms, "
            f"ranking={ranking_time:.1f}ms, total={total_time:.1f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Ranking error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def rank_with_strategy(strategy_name: str, candidates: List, 
                      session_state: Dict) -> tuple:
    """Rank candidates using specified strategy"""
    strategies = app_state['decision_layer']
    
    try:
        if strategy_name == 'bandit':
            bandit = strategies['bandit']['bandit']
            action_idx = bandit.select_action(session_state)
            weights = bandit.actions[action_idx]
            
            decision_layer = strategies['bandit']['decision_layer']
            decision_layer.fixed_weights = weights
            ranked_list = decision_layer.rank(candidates, session_state)
        else:
            strategy = strategies[strategy_name]
            ranked_list = strategy.rank(candidates, session_state)
            weights = strategy.get_current_weights()
        
        return ranked_list, weights
        
    except Exception as e:
        logger.error(f"Strategy error: {e}, falling back to baseline")
        # Fallback to baseline
        strategy = strategies['baseline']
        ranked_list = strategy.rank(candidates, session_state)
        weights = strategy.get_current_weights()
        return ranked_list, weights


def cold_start_fallback(request: RankRequest, start_time: float) -> RankResponse:
    """Fallback for cold start users"""
    logger.warning(f"Cold start for user {request.user_id}")
    
    # Return popular + diverse articles
    # In production, this would query a pre-computed popular set
    articles = [
        ArticleResponse(
            article_id=f"popular_{i}",
            title=f"Popular Article {i}",
            category="general",
            score=1.0 - i * 0.05,
            ctr_score=0.5,
            dwell_score=0.5,
            retention_score=0.5
        )
        for i in range(request.k)
    ]
    
    return RankResponse(
        articles=articles,
        weights=[0.4, 0.3, 0.2, 0.1],
        strategy='cold_start',
        latency_ms=(time.time() - start_time) * 1000
    )


# Logging endpoint
@app.post("/log")
async def log_interaction(request: Request):
    """Log user interaction for bandit learning"""
    try:
        data = await request.json()
        
        # Store in Redis for batch processing
        redis_client = app_state.get('redis_client')
        if redis_client:
            redis_client.lpush('interaction_logs', json.dumps(data))
        
        return {"status": "logged"}
        
    except Exception as e:
        logger.error(f"Logging error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": all(k in app_state for k in ['retrieval', 'ranking', 'decision_layer']),
        "redis_connected": app_state.get('redis_client') is not None
    }


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    redis_client = app_state.get('redis_client')
    
    metrics = {
        "models_loaded": True,
        "redis_connected": redis_client is not None
    }
    
    if redis_client:
        try:
            info = redis_client.info()
            metrics['redis_memory_mb'] = info['used_memory'] / 1024 / 1024
            metrics['redis_keys'] = redis_client.dbsize()
        except Exception as e:
            logger.error(f"Metrics error: {e}")
    
    return metrics


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
