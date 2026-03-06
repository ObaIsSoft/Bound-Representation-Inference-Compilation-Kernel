"""
Production UserAgent - User Management & Permissions

Follows BRICK OS patterns:
- NO hardcoded roles - uses database-driven RBAC
- OAuth/OIDC integration
- Audit logging
- Organization multi-tenancy
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Role(Enum):
    """User roles."""
    ADMIN = "admin"
    DESIGNER = "designer"
    ENGINEER = "engineer"
    VIEWER = "viewer"
    API = "api"


class Permission(Enum):
    """System permissions."""
    CREATE_PROJECT = "create:project"
    READ_PROJECT = "read:project"
    UPDATE_PROJECT = "update:project"
    DELETE_PROJECT = "delete:project"
    RUN_SIMULATION = "run:simulation"
    EXPORT_DESIGN = "export:design"
    MANAGE_USERS = "manage:users"
    MANAGE_ORG = "manage:organization"


@dataclass
class User:
    """User record."""
    id: str
    email: str
    name: str
    roles: List[Role]
    organization_id: str
    permissions: Set[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


class UserAgent:
    """
    Production user management agent.
    
    Manages:
    - User authentication
    - Role-based access control (RBAC)
    - Organization membership
    - Permission verification
    - Audit logging
    
    FAIL FAST: Returns error if user not found in database.
    """
    
    def __init__(self):
        self.name = "UserAgent"
        self._initialized = False
        self.supabase = None
        self._role_permissions: Dict[str, Set[Permission]] = {}
        
    async def initialize(self):
        """Initialize database connection and load role permissions."""
        if self._initialized:
            return
        
        try:
            from backend.services import supabase_service
            self.supabase = supabase_service.supabase
            await self._load_role_permissions()
            self._initialized = True
            logger.info("UserAgent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise RuntimeError(f"UserAgent initialization failed: {e}")
    
    async def _load_role_permissions(self):
        """Load role-permission mappings from database."""
        try:
            result = await self.supabase.table("role_permissions")\
                .select("role, permission")\
                .execute()
            
            self._role_permissions = {}
            for row in result.data:
                role = row.get("role")
                permission = row.get("permission")
                if role and permission:
                    if role not in self._role_permissions:
                        self._role_permissions[role] = set()
                    try:
                        self._role_permissions[role].add(Permission(permission))
                    except ValueError:
                        logger.warning(f"Unknown permission: {permission}")
        except Exception as e:
            logger.error(f"Failed to load role permissions: {e}")
            raise RuntimeError(f"Cannot load role permissions from database: {e}")
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute user management operation.
        
        Args:
            params: {
                "operation": "authenticate" | "authorize" | "get_user" | "list_users",
                "user_id": "...",
                "permission": "...",
                "resource": "..."
            }
        """
        await self.initialize()
        
        operation = params.get("operation", "get_user")
        
        if operation == "authenticate":
            return await self._authenticate(params)
        
        elif operation == "authorize":
            return await self._authorize(params)
        
        elif operation == "get_user":
            return await self._get_user(params.get("user_id"))
        
        elif operation == "list_users":
            return await self._list_users(params.get("organization_id"))
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _authenticate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate user via OAuth/OIDC."""
        
        token = params.get("access_token")
        provider = params.get("provider", "supabase")
        
        if not token:
            raise ValueError("Access token required for authentication")
        
        try:
            # Verify token with Supabase
            result = await self.supabase.auth.get_user(token)
            
            if result.user:
                user = await self._get_or_create_user(result.user)
                return {
                    "status": "authenticated",
                    "user": self._user_to_dict(user)
                }
            else:
                raise ValueError("Invalid token")
        
        except Exception as e:
            raise ValueError(f"Authentication failed: {e}")
    
    async def _authorize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if user has permission for resource."""
        
        user_id = params.get("user_id")
        permission_str = params.get("permission")
        resource = params.get("resource")
        
        if not user_id or not permission_str:
            raise ValueError("user_id and permission required")
        
        try:
            permission = Permission(permission_str)
        except ValueError:
            raise ValueError(f"Invalid permission: {permission_str}")
        
        # Get user
        user = await self._get_user_obj(user_id)
        
        # Check permission
        has_permission = permission in user.permissions
        
        # Log access attempt
        await self._log_access(user_id, permission.value, resource, has_permission)
        
        return {
            "authorized": has_permission,
            "user_id": user_id,
            "permission": permission.value,
            "resource": resource,
            "reason": None if has_permission else "Insufficient permissions"
        }
    
    async def _get_user(self, user_id: str) -> Dict[str, Any]:
        """Get user by ID."""
        
        user = await self._get_user_obj(user_id)
        return {
            "status": "found",
            "user": self._user_to_dict(user)
        }
    
    async def _get_user_obj(self, user_id: str) -> User:
        """Get user object from database."""
        
        try:
            result = await self.supabase.table("users")\
                .select("*")\
                .eq("id", user_id)\
                .single()\
                .execute()
            
            if not result.data:
                raise ValueError(f"User {user_id} not found")
            
            data = result.data
            
            # Parse roles from database
            roles_data = data.get("roles", [])
            if isinstance(roles_data, str):
                roles_data = [roles_data]
            
            roles = []
            for r in roles_data:
                try:
                    roles.append(Role(r))
                except ValueError:
                    logger.warning(f"Unknown role: {r}")
            
            if not roles:
                raise ValueError(f"User {user_id} has no valid roles assigned")
            
            # Calculate permissions from roles
            permissions = set()
            for role in roles:
                role_perms = self._role_permissions.get(role.value, set())
                permissions.update(role_perms)
            
            created_at_str = data.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                except ValueError:
                    created_at = datetime.now()
            else:
                created_at = datetime.now()
            
            last_login_str = data.get("last_login")
            last_login = None
            if last_login_str:
                try:
                    last_login = datetime.fromisoformat(last_login_str.replace('Z', '+00:00'))
                except ValueError:
                    pass
            
            return User(
                id=data["id"],
                email=data["email"],
                name=data.get("name", ""),
                roles=roles,
                organization_id=data.get("organization_id", ""),
                permissions=permissions,
                created_at=created_at,
                last_login=last_login,
                is_active=data.get("is_active", True)
            )
        
        except Exception as e:
            if "not found" in str(e).lower():
                raise
            raise ValueError(f"Could not retrieve user: {e}")
    
    async def _get_or_create_user(self, auth_user: Any) -> User:
        """Get existing user or create new from auth provider."""
        
        try:
            return await self._get_user_obj(auth_user.id)
        except ValueError:
            # Get default role from database
            try:
                default_role_result = await self.supabase.table("default_settings")\
                    .select("default_user_role")\
                    .single()\
                    .execute()
                default_role = default_role_result.data.get("default_user_role", "viewer") if default_role_result.data else "viewer"
            except Exception:
                default_role = "viewer"
            
            # Create new user
            user_data = {
                "id": auth_user.id,
                "email": auth_user.email,
                "name": auth_user.user_metadata.get("name", "") if hasattr(auth_user, 'user_metadata') else "",
                "roles": [default_role],
                "organization_id": "default",
                "created_at": datetime.now().isoformat(),
                "is_active": True
            }
            
            try:
                await self.supabase.table("users").insert(user_data).execute()
                logger.info(f"Created new user: {auth_user.email}")
                return await self._get_user_obj(auth_user.id)
            except Exception as e:
                raise ValueError(f"Could not create user: {e}")
    
    async def _list_users(self, organization_id: Optional[str]) -> Dict[str, Any]:
        """List users in organization."""
        
        try:
            query = self.supabase.table("users").select("*")
            if organization_id:
                query = query.eq("organization_id", organization_id)
            
            result = await query.execute()
            
            users = []
            for data in result.data:
                users.append({
                    "id": data["id"],
                    "email": data["email"],
                    "name": data.get("name", ""),
                    "roles": data.get("roles", []),
                    "is_active": data.get("is_active", True)
                })
            
            return {
                "status": "success",
                "count": len(users),
                "users": users
            }
        
        except Exception as e:
            raise ValueError(f"Could not list users: {e}")
    
    async def _log_access(
        self,
        user_id: str,
        permission: str,
        resource: Optional[str],
        granted: bool
    ):
        """Log access attempt to audit log."""
        
        try:
            await self.supabase.table("audit_log").insert({
                "user_id": user_id,
                "action": permission,
                "resource": resource,
                "granted": granted,
                "timestamp": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.warning(f"Could not write audit log: {e}")
    
    def _user_to_dict(self, user: User) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "roles": [r.value for r in user.roles],
            "permissions": [p.value for p in user.permissions],
            "organization_id": user.organization_id,
            "is_active": user.is_active,
            "last_login": user.last_login.isoformat() if user.last_login else None
        }


# Convenience functions
async def check_permission(user_id: str, permission: str, resource: str = None) -> bool:
    """Quick permission check."""
    agent = UserAgent()
    result = await agent.run({
        "operation": "authorize",
        "user_id": user_id,
        "permission": permission,
        "resource": resource
    })
    return result.get("authorized", False)


async def get_user_info(user_id: str) -> Dict[str, Any]:
    """Get user information."""
    agent = UserAgent()
    result = await agent.run({
        "operation": "get_user",
        "user_id": user_id
    })
    return result.get("user", {})
