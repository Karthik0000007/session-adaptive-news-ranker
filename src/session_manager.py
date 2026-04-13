"""
Session Manager

Handles session state persistence and updates using Redis.
Fix 5 (P1): Uses JSON serialization instead of pickle for security.
"""

import redis
import json
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user session state with Redis backend (JSON serialization)"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379,
                 redis_db: int = 0, ttl: int = 3600):
        """
        Initialize session manager
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            ttl: Session TTL in seconds (default 1 hour)
        """
        self.ttl = ttl
        self.redis_client = self._init_redis(redis_host, redis_port, redis_db)
        
    def _init_redis(self, host: str, port: int, db: int) -> Optional[redis.Redis]:
        """Initialize Redis client"""
        try:
            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True  # Fix 5: string mode for JSON
            )
            client.ping()
            logger.info("Redis connected successfully")
            return client
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}")
            return None
    
    def get_session(self, user_id: str) -> Optional[Dict]:
        """
        Get session state for user
        
        Args:
            user_id: User identifier
            
        Returns:
            Session state dict or None if not found
        """
        if not self.redis_client:
            return None
        
        try:
            key = f"session:{user_id}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)  # Fix 5: JSON instead of pickle
        except Exception as e:
            logger.error(f"Error getting session: {e}")
        
        return None
    
    def update_session(self, user_id: str, state: Dict):
        """
        Update session state for user
        
        Args:
            user_id: User identifier
            state: Session state dict
        """
        if not self.redis_client:
            return
        
        try:
            key = f"session:{user_id}"
            self.redis_client.setex(
                key,
                self.ttl,
                json.dumps(state)  # Fix 5: JSON instead of pickle
            )
        except Exception as e:
            logger.error(f"Error updating session: {e}")
    
    def delete_session(self, user_id: str):
        """
        Delete session for user (APPI Article 30 support)
        
        Args:
            user_id: User identifier
        """
        if not self.redis_client:
            return
        
        try:
            key = f"session:{user_id}"
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
    
    def get_active_sessions(self) -> int:
        """Get count of active sessions"""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys("session:*")
            return len(keys)
        except Exception as e:
            logger.error(f"Error counting sessions: {e}")
            return 0
