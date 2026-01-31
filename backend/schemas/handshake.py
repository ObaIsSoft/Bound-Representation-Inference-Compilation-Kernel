"""
ISA/Schema Handshake Protocol

This module implements the handshake protocol for ISA (Instruction Set Architecture)
version negotiation and schema validation between frontend and backend.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ISAVersion(BaseModel):
    """ISA version information"""
    major: int
    minor: int
    patch: int
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    @classmethod
    def from_string(cls, version_str: str) -> "ISAVersion":
        """Parse version string like '1.0.0'"""
        parts = version_str.split(".")
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0
        )


class PodSchema(BaseModel):
    """Schema for a single ISA pod (component)"""
    id: str
    name: str
    type: str  # e.g., "structural", "propulsion", "electronics"
    parameters: Dict[str, Any]
    constraints: Dict[str, Any]
    children: List[str] = []  # Child pod IDs
    parent: Optional[str] = None  # Parent pod ID
    metadata: Dict[str, Any] = {}


class ISAHierarchy(BaseModel):
    """Complete ISA hierarchy tree"""
    root_pod_id: str
    pods: Dict[str, PodSchema]  # pod_id -> PodSchema
    version: ISAVersion
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class HandshakeRequest(BaseModel):
    """Client handshake request"""
    client_version: str  # ISA version client supports
    client_id: str  # Unique client identifier
    requested_features: List[str] = []  # Optional feature flags
    timestamp: datetime = Field(default_factory=datetime.now)


class HandshakeResponse(BaseModel):
    """Server handshake response"""
    server_version: str  # ISA version server supports
    negotiated_version: str  # Version both can use
    compatible: bool  # Whether versions are compatible
    isa_hierarchy: Optional[ISAHierarchy] = None  # Full ISA tree
    supported_features: List[str] = []  # Features server supports
    session_id: str  # Unique session identifier
    timestamp: datetime = Field(default_factory=datetime.now)
    message: str = ""  # Human-readable message


class ExecutionContext(BaseModel):
    """Execution context with focused pod"""
    session_id: str
    focused_pod_id: Optional[str] = None  # Current pod in focus
    scope: str = "global"  # "global" or "scoped"
    parameters: Dict[str, Any] = {}


# Current ISA version
CURRENT_ISA_VERSION = ISAVersion(major=1, minor=0, patch=0)


def is_compatible(client_version: ISAVersion, server_version: ISAVersion) -> bool:
    """
    Check if client and server versions are compatible.
    
    Rules:
    - Major versions must match
    - Server minor version must be >= client minor version
    """
    if client_version.major != server_version.major:
        return False
    
    if server_version.minor < client_version.minor:
        return False
    
    return True


def negotiate_version(client_version: ISAVersion, server_version: ISAVersion) -> ISAVersion:
    """
    Negotiate the best version both client and server can use.
    
    Returns the highest compatible version.
    """
    if not is_compatible(client_version, server_version):
        raise ValueError(f"Incompatible versions: client={client_version}, server={server_version}")
    
    # Use the lower minor version
    return ISAVersion(
        major=server_version.major,
        minor=min(client_version.minor, server_version.minor),
        patch=0  # Always use patch 0 for negotiated version
    )
