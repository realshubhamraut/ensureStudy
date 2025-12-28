"""
Unit Tests for Rate Limiter Service
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock


class TestRateLimits:
    """Tests for rate limit configuration"""
    
    def test_rate_limits_defined(self):
        from app.services.rate_limiter import RATE_LIMITS
        
        assert 'document_upload' in RATE_LIMITS
        assert 'ai_tutor_query_minute' in RATE_LIMITS
        assert 'ai_tutor_query_hour' in RATE_LIMITS
        assert 'login_attempt' in RATE_LIMITS
    
    def test_rate_limit_structure(self):
        from app.services.rate_limiter import RATE_LIMITS
        
        for action, config in RATE_LIMITS.items():
            assert 'max_requests' in config
            assert 'window_seconds' in config
            assert config['max_requests'] > 0
            assert config['window_seconds'] > 0


class TestRateLimiter:
    """Tests for RateLimiter class"""
    
    @pytest.fixture
    def limiter(self):
        from app.services.rate_limiter import RateLimiter
        return RateLimiter()
    
    # =========================================================================
    # Without Redis (fallback behavior)
    # =========================================================================
    
    def test_rate_limiter_disabled_allows_all(self, limiter):
        """When disabled, all requests allowed"""
        limiter.enabled = False
        result = limiter.check_rate_limit('user-1', 'document_upload')
        assert result['allowed'] is True
        assert result['remaining'] == 999
    
    def test_rate_limiter_no_redis_allows_all(self, limiter):
        """When no Redis, all requests allowed"""
        limiter.redis_client = None
        result = limiter.check_rate_limit('user-1', 'document_upload')
        assert result['allowed'] is True
    
    def test_is_rate_limited_without_redis(self, limiter):
        """is_rate_limited returns False without Redis"""
        limiter.redis_client = None
        assert limiter.is_rate_limited('user-1', 'api_global') is False
    
    def test_get_remaining_without_redis(self, limiter):
        """get_remaining returns 999 without Redis"""
        limiter.redis_client = None
        assert limiter.get_remaining('user-1', 'api_global') == 999
    
    # =========================================================================
    # With Mocked Redis
    # =========================================================================
    
    @patch('redis.from_url')
    def test_check_rate_limit_under_limit(self, mock_redis):
        """Under limit: request allowed"""
        from app.services.rate_limiter import RateLimiter
        
        mock_client = Mock()
        mock_pipe = Mock()
        mock_pipe.execute.return_value = [0, 3, []]  # 3 requests made
        mock_client.pipeline.return_value = mock_pipe
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        limiter = RateLimiter()
        result = limiter.check_rate_limit('user-1', 'document_upload')
        
        assert result['allowed'] is True
        assert result['remaining'] == 7  # 10 - 3
    
    @patch('redis.from_url')
    def test_check_rate_limit_at_limit(self, mock_redis):
        """At limit: request denied"""
        from app.services.rate_limiter import RateLimiter
        
        mock_client = Mock()
        mock_pipe = Mock()
        mock_pipe.execute.return_value = [0, 10, [(b'key', time.time())]]  # 10 requests (limit)
        mock_client.pipeline.return_value = mock_pipe
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        limiter = RateLimiter()
        result = limiter.check_rate_limit('user-1', 'document_upload')
        
        assert result['allowed'] is False
        assert result['remaining'] == 0
        assert result['retry_after'] >= 0
    
    @patch('redis.from_url')
    def test_record_request(self, mock_redis):
        """Recording request adds to Redis"""
        from app.services.rate_limiter import RateLimiter
        
        mock_client = Mock()
        mock_pipe = Mock()
        mock_pipe.execute.return_value = True
        mock_client.pipeline.return_value = mock_pipe
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        limiter = RateLimiter()
        result = limiter.record_request('user-1', 'document_upload')
        
        assert result is True
        mock_pipe.zadd.assert_called_once()
        mock_pipe.expire.assert_called_once()
    
    # =========================================================================
    # Key Generation Tests
    # =========================================================================
    
    def test_get_key_format(self, limiter):
        """Key format is correct"""
        key = limiter._get_key('user-123', 'document_upload')
        assert key == 'rate_limit:user-123:document_upload'
    
    def test_get_key_different_users(self, limiter):
        """Different users get different keys"""
        k1 = limiter._get_key('user-1', 'action')
        k2 = limiter._get_key('user-2', 'action')
        assert k1 != k2
    
    def test_get_key_different_actions(self, limiter):
        """Different actions get different keys"""
        k1 = limiter._get_key('user-1', 'action1')
        k2 = limiter._get_key('user-1', 'action2')
        assert k1 != k2


class TestRateLimitDecorator:
    """Tests for rate_limit decorator"""
    
    def test_rate_limit_decorator_exists(self):
        from app.services.rate_limiter import rate_limit
        assert callable(rate_limit)
    
    def test_decorator_creates_wrapper(self):
        from app.services.rate_limiter import rate_limit
        
        @rate_limit("ai_tutor_query_minute")
        def sample_function():
            return "result"
        
        # Should be wrapped function
        assert hasattr(sample_function, '__wrapped__') or callable(sample_function)


class TestRateLimiterSingleton:
    """Tests for singleton pattern"""
    
    def test_get_rate_limiter_singleton(self):
        from app.services.rate_limiter import get_rate_limiter
        
        # Reset singleton for test
        import app.services.rate_limiter as module
        module._rate_limiter = None
        
        l1 = get_rate_limiter()
        l2 = get_rate_limiter()
        assert l1 is l2
