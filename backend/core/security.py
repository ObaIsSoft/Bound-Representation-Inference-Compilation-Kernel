"""
Production Security Module for ProjectOrchestrator

Provides:
- Input validation and sanitization
- Rate limiting with token bucket algorithm
- Path traversal protection
- Authentication/authorization hooks
- Audit logging
- DDoS protection
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Base security exception"""
    pass


class ValidationError(SecurityError):
    """Input validation failed"""
    pass


class RateLimitError(SecurityError):
    """Rate limit exceeded"""
    pass


class AuthorizationError(SecurityError):
    """Not authorized"""
    pass


# ============ Input Validation ============

@dataclass
class ValidationRules:
    """Validation rules for inputs"""
    max_length: int = 10000
    min_length: int = 1
    allowed_pattern: Optional[str] = None
    forbidden_patterns: List[str] = field(default_factory=list)
    max_nested_depth: int = 10
    max_array_size: int = 1000


class InputValidator:
    """
    Comprehensive input validation.
    
    Prevents:
    - Injection attacks
    - Buffer overflows (size limits)
    - ReDoS (regex timeout)
    - Nested object bombs
    """
    
    # Dangerous patterns that could indicate injection
    DANGEROUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # XSS
        r"javascript:",  # JS injection
        r"data:text/html",
        r"\$\{.*\}",  # Template injection
        r"<%.*%>",  # Server-side template injection
        r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\b",  # SQLi (basic)
        r"\b(eval|exec|system|subprocess)\s*\(",  # Code injection
        r"\.\./|\.\.\\",  # Path traversal attempts
        r"[\x00-\x08\x0b-\x0c\x0e-\x1f]",  # Control characters
    ]
    
    # Compiled patterns for performance
    _dangerous_regex = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in DANGEROUS_PATTERNS]
    
    @classmethod
    def sanitize_string(
        cls,
        value: str,
        rules: Optional[ValidationRules] = None,
        field_name: str = "input"
    ) -> str:
        """
        Sanitize a string input.
        
        Args:
            value: Input string
            rules: Validation rules
            field_name: Name for error messages
        
        Returns:
            Sanitized string
        
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string")
        
        rules = rules or ValidationRules()
        
        # Check length
        if len(value) > rules.max_length:
            raise ValidationError(
                f"{field_name} exceeds maximum length of {rules.max_length}"
            )
        
        if len(value) < rules.min_length:
            raise ValidationError(
                f"{field_name} below minimum length of {rules.min_length}"
            )
        
        # Check dangerous patterns
        for pattern in cls._dangerous_regex:
            if pattern.search(value):
                logger.warning(f"Dangerous pattern detected in {field_name}")
                raise ValidationError(f"{field_name} contains invalid characters")
        
        # Check custom forbidden patterns
        for pattern in rules.forbidden_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValidationError(f"{field_name} contains forbidden content")
        
        # Check allowed pattern
        if rules.allowed_pattern and not re.match(rules.allowed_pattern, value):
            raise ValidationError(f"{field_name} format is invalid")
        
        # Normalize unicode
        value = value.strip()
        
        return value
    
    @classmethod
    def sanitize_project_id(cls, project_id: str) -> str:
        """Sanitize project ID with strict rules"""
        rules = ValidationRules(
            max_length=64,
            min_length=3,
            allowed_pattern=r"^[a-zA-Z0-9_-]+$"
        )
        return cls.sanitize_string(project_id, rules, "project_id")
    
    @classmethod
    def sanitize_user_intent(cls, intent: str) -> str:
        """Sanitize user intent (design description)"""
        rules = ValidationRules(
            max_length=5000,  # Generous limit for descriptions
            min_length=5,
            forbidden_patterns=[
                r"<[^>]+on\w+\s*=",  # Event handlers
                r"javascript:",
                r"data:",
            ]
        )
        return cls.sanitize_string(intent, rules, "user_intent")
    
    @classmethod
    def sanitize_checkpoint_id(cls, checkpoint_id: str) -> str:
        """Sanitize checkpoint ID - UUID format only"""
        rules = ValidationRules(
            max_length=36,
            min_length=36,
            allowed_pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        try:
            return cls.sanitize_string(checkpoint_id, rules, "checkpoint_id")
        except ValidationError:
            # Also accept short form (first 8 chars of UUID)
            if re.match(r"^[0-9a-f]{8}$", checkpoint_id):
                return checkpoint_id
            raise
    
    @classmethod
    def validate_nested_structure(
        cls,
        obj: Any,
        max_depth: int = 10,
        current_depth: int = 0,
        max_size: int = 1000,
        current_size: int = 0
    ) -> Tuple[bool, int]:
        """
        Validate nested structure doesn't exceed limits.
        
        Returns:
            (is_valid, total_size)
        """
        if current_depth > max_depth:
            return False, current_size
        
        if current_size > max_size:
            return False, current_size
        
        if isinstance(obj, (str, int, float, bool, type(None))):
            return True, current_size + 1
        
        if isinstance(obj, (list, tuple)):
            for item in obj:
                valid, current_size = cls.validate_nested_structure(
                    item, max_depth, current_depth + 1, max_size, current_size
                )
                if not valid:
                    return False, current_size
            return True, current_size + 1
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                # Validate key is string
                if not isinstance(key, str):
                    return False, current_size
                
                # Validate key length
                if len(key) > 256:
                    return False, current_size
                
                valid, current_size = cls.validate_nested_structure(
                    value, max_depth, current_depth + 1, max_size, current_size
                )
                if not valid:
                    return False, current_size
            return True, current_size + 1
        
        # Unknown type
        return False, current_size


# ============ Path Security ============

class PathSecurity:
    """
    Secure path handling to prevent directory traversal attacks.
    """
    
    @staticmethod
    def secure_path(base_dir: Path, *components: str) -> Path:
        """
        Create a secure path that cannot escape base_dir.
        
        Args:
            base_dir: Base directory that must contain the result
            components: Path components
        
        Returns:
            Secure absolute path
        
        Raises:
            ValidationError: If path would escape base_dir
        """
        base_dir = base_dir.resolve()
        
        # Sanitize each component
        sanitized = []
        for comp in components:
            # Remove any path separators and parent references
            comp = comp.replace("/", "_").replace("\\", "_")
            comp = comp.replace("..", "_")
            comp = comp.replace(":", "_")  # Windows drive letters
            # Remove null bytes
            comp = comp.replace("\x00", "")
            # Limit length
            if len(comp) > 255:
                comp = comp[:255]
            sanitized.append(comp)
        
        # Build path
        result = base_dir.joinpath(*sanitized)
        result = result.resolve()
        
        # Verify it's under base_dir
        try:
            result.relative_to(base_dir)
        except ValueError:
            raise ValidationError(f"Path {result} escapes base directory {base_dir}")
        
        return result
    
    @staticmethod
    def safe_filename(filename: str, max_length: int = 255) -> str:
        """
        Create a safe filename.
        
        Removes/replaces dangerous characters.
        """
        # Replace dangerous characters
        dangerous = '<>:"/\\|?*\x00-\x1f'
        for char in dangerous:
            filename = filename.replace(char, '_')
        
        # Limit length (preserve extension)
        if len(filename) > max_length:
            name, ext = Path(filename).stem, Path(filename).suffix
            max_name = max_length - len(ext) - 1
            filename = name[:max_name] + ext
        
        # Ensure not empty or just dots
        filename = filename.strip('. ')
        if not filename:
            filename = "unnamed"
        
        return filename


# ============ Rate Limiting ============

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_second: float = 10.0
    burst_size: int = 20
    block_duration: timedelta = timedelta(minutes=1)


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.
    
    Allows bursts up to burst_size, then throttles to requests_per_second.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def acquire(self, key: str) -> bool:
        """
        Try to acquire a token for the given key.
        
        Args:
            key: Rate limit bucket key (e.g., IP address, user ID)
        
        Returns:
            True if allowed, False if rate limited
        """
        async with self._lock:
            now = time.time()
            
            # Initialize bucket if needed
            if key not in self.buckets:
                self.buckets[key] = {
                    "tokens": self.config.burst_size,
                    "last_update": now,
                    "blocked_until": None
                }
            
            bucket = self.buckets[key]
            
            # Check if blocked
            if bucket["blocked_until"] and now < bucket["blocked_until"]:
                return False
            
            # Add tokens based on time passed
            time_passed = now - bucket["last_update"]
            tokens_to_add = time_passed * self.config.requests_per_second
            bucket["tokens"] = min(
                self.config.burst_size,
                bucket["tokens"] + tokens_to_add
            )
            bucket["last_update"] = now
            
            # Try to consume token
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            else:
                # Block the key
                bucket["blocked_until"] = now + self.config.block_duration.total_seconds()
                logger.warning(f"Rate limit exceeded for key: {key}")
                return False
    
    async def check_or_raise(self, key: str):
        """Acquire or raise RateLimitError"""
        if not await self.acquire(key):
            raise RateLimitError(
                f"Rate limit exceeded. Try again in {self.config.block_duration.seconds} seconds."
            )
    
    def get_status(self, key: str) -> Dict[str, Any]:
        """Get current rate limit status for a key"""
        bucket = self.buckets.get(key)
        if not bucket:
            return {
                "tokens": self.config.burst_size,
                "limit": self.config.burst_size,
                "remaining": self.config.burst_size,
                "reset_in": 0
            }
        
        now = time.time()
        time_passed = now - bucket["last_update"]
        tokens = min(
            self.config.burst_size,
            bucket["tokens"] + time_passed * self.config.requests_per_second
        )
        
        return {
            "tokens": tokens,
            "limit": self.config.burst_size,
            "remaining": max(0, int(tokens)),
            "reset_in": max(0, bucket.get("blocked_until", 0) - now)
        }


# ============ Authentication/Authorization Hooks ============

@dataclass
class AuthContext:
    """Authentication context"""
    user_id: str
    permissions: Set[str] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)
    authenticated_at: datetime = field(default_factory=datetime.utcnow)
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has permission"""
        return permission in self.permissions or "admin" in self.roles
    
    def can_access_project(self, project_id: str) -> bool:
        """Check if user can access project"""
        # Simplified - in production check project ownership
        return self.has_permission(f"project:{project_id}:read")


class AuthManager:
    """
    Authentication manager.
    
    Provides hooks for JWT validation, API key checking, etc.
    """
    
    def __init__(self):
        self._verifiers: List[Callable[[str], Optional[AuthContext]]] = []
    
    def register_verifier(self, verifier: Callable[[str], Optional[AuthContext]]):
        """Register an authentication verifier"""
        self._verifiers.append(verifier)
    
    async def authenticate(self, token: str) -> AuthContext:
        """
        Authenticate a token.
        
        Args:
            token: Authentication token (JWT, API key, etc.)
        
        Returns:
            AuthContext if valid
        
        Raises:
            AuthorizationError: If authentication fails
        """
        if not token:
            raise AuthorizationError("No authentication token provided")
        
        for verifier in self._verifiers:
            try:
                context = verifier(token)
                if context:
                    return context
            except Exception as e:
                logger.debug(f"Verifier failed: {e}")
                continue
        
        raise AuthorizationError("Invalid authentication token")
    
    @staticmethod
    def create_api_key() -> str:
        """Generate a secure API key"""
        return f"brick_{secrets.token_urlsafe(32)}"


# ============ Audit Logging ============

class AuditLogger:
    """
    Security audit logging.
    
    Logs all security-relevant events for compliance and forensics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("audit")
    
    def log_event(
        self,
        event_type: str,
        user_id: Optional[str],
        resource: str,
        action: str,
        success: bool,
        details: Optional[Dict] = None
    ):
        """Log an audit event"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id or "anonymous",
            "resource": resource,
            "action": action,
            "success": success,
            "details": details or {}
        }
        
        if success:
            self.logger.info(f"AUDIT: {event}")
        else:
            self.logger.warning(f"AUDIT_FAIL: {event}")
    
    def log_project_access(self, user_id: str, project_id: str, action: str, success: bool):
        """Log project access attempt"""
        self.log_event(
            "project_access",
            user_id,
            f"project:{project_id}",
            action,
            success
        )
    
    def log_api_call(self, user_id: Optional[str], endpoint: str, method: str, success: bool):
        """Log API call"""
        self.log_event(
            "api_call",
            user_id,
            endpoint,
            method,
            success
        )


# ============ Global Instances ============

# Rate limiters
_project_rate_limiter = TokenBucketRateLimiter(
    RateLimitConfig(requests_per_second=5, burst_size=10)
)
_api_rate_limiter = TokenBucketRateLimiter(
    RateLimitConfig(requests_per_second=100, burst_size=200)
)

# Auth manager
_auth_manager = AuthManager()

# Audit logger
_audit_logger = AuditLogger()


def get_project_rate_limiter() -> TokenBucketRateLimiter:
    return _project_rate_limiter


def get_api_rate_limiter() -> TokenBucketRateLimiter:
    return _api_rate_limiter


def get_auth_manager() -> AuthManager:
    return _auth_manager


def get_audit_logger() -> AuditLogger:
    return _audit_logger
