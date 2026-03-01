"""
FIX-004: Directory Creation Safeguards

Provides safe, atomic directory operations with:
- Automatic parent directory creation
- Permission checks
- Race condition handling
- Logging of all operations
- Cleanup on failure
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional, Union, List
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class DirectoryError(Exception):
    """Base exception for directory operations"""
    pass


class DirectoryPermissionError(DirectoryError):
    """Permission denied for directory operation"""
    pass


class DirectoryExistsError(DirectoryError):
    """Directory already exists (when exclusive creation requested)"""
    pass


class DirectoryManager:
    """
    Thread-safe directory manager with safeguards.
    
    Features:
    - Atomic directory creation
    - Permission validation
    - Automatic cleanup on failure
    - Disk space checks
    - Path traversal prevention
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._created_paths: List[Path] = []
    
    def ensure_directory(
        self,
        path: Union[str, Path],
        mode: int = 0o755,
        parents: bool = True,
        exist_ok: bool = True,
        check_writable: bool = True
    ) -> Path:
        """
        Safely ensure a directory exists with full safeguards.
        
        Args:
            path: Directory path (relative to base_path or absolute)
            mode: Directory permissions (default 0o755)
            parents: Create parent directories if needed
            exist_ok: Don't raise error if directory exists
            check_writable: Verify directory is writable
            
        Returns:
            Resolved Path object
            
        Raises:
            DirectoryPermissionError: If permission denied
            DirectoryExistsError: If exists_ok=False and directory exists
            DirectoryError: For other failures
        """
        target = self._resolve_path(path)
        
        # Check for path traversal attacks
        try:
            target.relative_to(self.base_path)
        except ValueError:
            if not target.is_absolute():
                logger.warning(f"Path traversal attempt detected: {path}")
                raise DirectoryError(f"Path traversal not allowed: {path}")
        
        # Check if already exists
        if target.exists():
            if not exist_ok:
                raise DirectoryExistsError(f"Directory already exists: {target}")
            if not target.is_dir():
                raise DirectoryError(f"Path exists but is not a directory: {target}")
            if check_writable and not os.access(target, os.W_OK):
                raise DirectoryPermissionError(f"Directory not writable: {target}")
            logger.debug(f"Directory already exists: {target}")
            return target
        
        # Create parent directories if needed
        if parents:
            parent = target.parent
            if not parent.exists():
                logger.debug(f"Creating parent directories: {parent}")
                try:
                    parent.mkdir(parents=True, mode=mode, exist_ok=True)
                except PermissionError as e:
                    raise DirectoryPermissionError(f"Cannot create parent directory {parent}: {e}")
        
        # Create the directory atomically using temp + rename pattern
        try:
            # Create with a temp name first
            temp_dir = tempfile.mkdtemp(prefix=f".tmp_{target.name}_", dir=target.parent)
            os.chmod(temp_dir, mode)
            
            # Atomic rename on Unix-like systems
            try:
                os.rename(temp_dir, target)
                logger.info(f"Created directory: {target}")
            except OSError:
                # Fallback: directory created by another process
                if not target.exists():
                    raise
                shutil.rmtree(temp_dir)
                
        except PermissionError as e:
            logger.error(f"Permission denied creating directory {target}: {e}")
            raise DirectoryPermissionError(f"Cannot create directory {target}: {e}")
        except Exception as e:
            logger.error(f"Failed to create directory {target}: {e}")
            raise DirectoryError(f"Failed to create directory {target}: {e}")
        
        # Verify writability
        if check_writable and not os.access(target, os.W_OK):
            raise DirectoryPermissionError(f"Directory created but not writable: {target}")
        
        self._created_paths.append(target)
        return target
    
    def create_project_directory(
        self,
        project_name: str,
        subdirs: Optional[List[str]] = None
    ) -> Path:
        """
        Create a complete project directory structure.
        
        Args:
            project_name: Name of the project (becomes directory name)
            subdirs: List of subdirectory names to create
            
        Returns:
            Path to project root
        """
        # Sanitize project name
        safe_name = self._sanitize_name(project_name)
        project_path = self.ensure_directory(safe_name)
        
        default_subdirs = [
            "models",
            "meshes",
            "simulations",
            "results",
            "exports",
            "cache",
            "logs"
        ]
        
        for subdir in (subdirs or default_subdirs):
            self.ensure_directory(project_path / subdir)
        
        logger.info(f"Created project structure: {project_path}")
        return project_path
    
    def cleanup_directory(self, path: Union[str, Path], confirm: bool = False) -> bool:
        """
        Safely remove a directory and all contents.
        
        Args:
            path: Directory to remove
            confirm: If True, only remove if created by this manager instance
            
        Returns:
            True if removed successfully
        """
        target = self._resolve_path(path)
        
        if confirm and target not in self._created_paths:
            logger.warning(f"Directory {target} not created by this manager, skipping cleanup")
            return False
        
        if not target.exists():
            return True
        
        try:
            shutil.rmtree(target)
            logger.info(f"Cleaned up directory: {target}")
            if target in self._created_paths:
                self._created_paths.remove(target)
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup directory {target}: {e}")
            return False
    
    def get_disk_space(self, path: Union[str, Path]) -> dict:
        """Get disk space information for a path"""
        target = self._resolve_path(path)
        
        try:
            stat = shutil.disk_usage(target)
            return {
                "total": stat.total,
                "used": stat.used,
                "free": stat.free,
                "total_gb": stat.total / (1024**3),
                "used_gb": stat.used / (1024**3),
                "free_gb": stat.free / (1024**3),
                "percent_used": (stat.used / stat.total) * 100
            }
        except Exception as e:
            logger.error(f"Cannot get disk space for {target}: {e}")
            raise DirectoryError(f"Cannot get disk space: {e}")
    
    def check_disk_space(
        self,
        path: Union[str, Path],
        required_bytes: int,
        safety_factor: float = 1.2
    ) -> bool:
        """
        Check if sufficient disk space is available.
        
        Args:
            path: Path to check
            required_bytes: Required space in bytes
            safety_factor: Multiply required space by this factor
            
        Returns:
            True if sufficient space available
        """
        space = self.get_disk_space(path)
        required_with_safety = int(required_bytes * safety_factor)
        
        if space["free"] < required_with_safety:
            logger.error(
                f"Insufficient disk space: {space['free_gb']:.2f} GB available, "
                f"{required_with_safety / (1024**3):.2f} GB required"
            )
            return False
        
        return True
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve path relative to base_path if not absolute"""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.base_path / path
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use as directory name"""
        # Remove potentially dangerous characters
        dangerous = ['/', '\\', '..', '~', '$', '%', '&', '*', '|', '<', '>', '?', '"', "'", '`']
        safe = name
        for char in dangerous:
            safe = safe.replace(char, '_')
        
        # Limit length
        max_len = 100
        if len(safe) > max_len:
            safe = safe[:max_len]
        
        # Ensure not empty
        if not safe or safe == '_':
            safe = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return safe


# Global directory manager instance
_dir_manager: Optional[DirectoryManager] = None


def get_directory_manager(base_path: Optional[Union[str, Path]] = None) -> DirectoryManager:
    """Get or create global directory manager instance"""
    global _dir_manager
    if _dir_manager is None or (base_path and Path(base_path) != _dir_manager.base_path):
        _dir_manager = DirectoryManager(base_path)
    return _dir_manager


@contextmanager
def temp_directory(prefix: str = "brick_", suffix: str = ""):
    """
    Context manager for temporary directory with automatic cleanup.
    
    Usage:
        with temp_directory() as tmpdir:
            # Use tmpdir
            pass
        # Automatically cleaned up
    """
    tmpdir = tempfile.mkdtemp(prefix=prefix, suffix=suffix)
    try:
        yield Path(tmpdir)
    finally:
        try:
            shutil.rmtree(tmpdir)
            logger.debug(f"Cleaned up temp directory: {tmpdir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory {tmpdir}: {e}")


def safe_makedirs(
    path: Union[str, Path],
    mode: int = 0o755,
    exist_ok: bool = True,
    check_writable: bool = True
) -> Path:
    """
    Drop-in replacement for os.makedirs with full safeguards.
    
    This is the primary function to use throughout the codebase.
    """
    manager = get_directory_manager()
    return manager.ensure_directory(
        path=path,
        mode=mode,
        parents=True,
        exist_ok=exist_ok,
        check_writable=check_writable
    )


# Convenience function for backward compatibility
def ensure_dir(path: Union[str, Path]) -> Path:
    """Simple directory creation with safeguards"""
    return safe_makedirs(path, exist_ok=True)
