"""
FIX-006: Async I/O Manager - Convert Blocking I/O to Async

Provides async wrappers for blocking I/O operations:
- File read/write
- Database operations
- External process execution
- Network requests

Uses ThreadPoolExecutor to offload blocking operations from the event loop.
"""

import asyncio
import aiofiles
import logging
from typing import Any, Callable, Optional, BinaryIO, TextIO, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import functools

logger = logging.getLogger(__name__)

# Global thread pool for blocking I/O operations
# Using a reasonable number of threads for I/O bound operations
_io_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="brick_io")


class AsyncIOManager:
    """
    Manager for async I/O operations.
    
    Converts blocking I/O to async using thread pool.
    """
    
    def __init__(self):
        self._executor = _io_executor
        self._loop = asyncio.get_event_loop()
    
    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a blocking function in the thread pool.
        
        Args:
            func: Blocking function to run
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of the function
        """
        # Use functools.partial for kwargs
        if kwargs:
            func = functools.partial(func, **kwargs)
        
        return await self._loop.run_in_executor(self._executor, func, *args)
    
    # -------------------------------------------------------------------------
    # File Operations
    # -------------------------------------------------------------------------
    
    async def read_text(self, path: Union[str, Path], 
                        encoding: str = "utf-8") -> str:
        """Async text file read"""
        path = Path(path)
        
        def _read():
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        
        return await self.run_in_thread(_read)
    
    async def write_text(self, path: Union[str, Path], 
                         content: str, 
                         encoding: str = "utf-8") -> None:
        """Async text file write"""
        path = Path(path)
        
        def _write():
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding=encoding) as f:
                f.write(content)
        
        await self.run_in_thread(_write)
    
    async def read_bytes(self, path: Union[str, Path]) -> bytes:
        """Async binary file read"""
        path = Path(path)
        
        def _read():
            with open(path, "rb") as f:
                return f.read()
        
        return await self.run_in_thread(_read)
    
    async def write_bytes(self, path: Union[str, Path], 
                          content: bytes) -> None:
        """Async binary file write"""
        path = Path(path)
        
        def _write():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(content)
        
        await self.run_in_thread(_write)
    
    async def file_exists(self, path: Union[str, Path]) -> bool:
        """Async file existence check"""
        path = Path(path)
        return await self.run_in_thread(path.exists)
    
    async def delete_file(self, path: Union[str, Path]) -> bool:
        """Async file deletion"""
        path = Path(path)
        
        def _delete():
            if path.exists():
                path.unlink()
                return True
            return False
        
        return await self.run_in_thread(_delete)
    
    # -------------------------------------------------------------------------
    # Directory Operations
    # -------------------------------------------------------------------------
    
    async def list_directory(self, path: Union[str, Path]) -> list:
        """Async directory listing"""
        path = Path(path)
        
        def _list():
            if not path.exists():
                return []
            return list(path.iterdir())
        
        return await self.run_in_thread(_list)
    
    async def walk_directory(self, path: Union[str, Path]):
        """Async directory walk (generator)"""
        path = Path(path)
        
        def _walk():
            for root, dirs, files in os.walk(path):
                yield root, dirs, files
        
        # Run in thread and collect results
        results = await self.run_in_thread(list, _walk())
        for item in results:
            yield item
    
    # -------------------------------------------------------------------------
    # JSON Operations
    # -------------------------------------------------------------------------
    
    async def read_json(self, path: Union[str, Path]) -> Any:
        """Async JSON file read"""
        import json
        
        content = await self.read_text(path)
        
        def _parse():
            return json.loads(content)
        
        return await self.run_in_thread(_parse)
    
    async def write_json(self, path: Union[str, Path], 
                         data: Any, 
                         indent: int = 2) -> None:
        """Async JSON file write"""
        import json
        
        def _serialize():
            return json.dumps(data, indent=indent, default=str)
        
        content = await self.run_in_thread(_serialize)
        await self.write_text(path, content)
    
    # -------------------------------------------------------------------------
    # CSV Operations
    # -------------------------------------------------------------------------
    
    async def read_csv(self, path: Union[str, Path]) -> list:
        """Async CSV file read"""
        import csv
        
        def _read():
            rows = []
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(row)
            return rows
        
        return await self.run_in_thread(_read)
    
    async def write_csv(self, path: Union[str, Path], 
                        rows: list,
                        headers: Optional[list] = None) -> None:
        """Async CSV file write"""
        import csv
        
        def _write():
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                if headers:
                    writer.writerow(headers)
                writer.writerows(rows)
        
        await self.run_in_thread(_write)


# ============================================================================
# DECORATOR FOR ASYNC CONVERSION
# ============================================================================

def async_io(func: Callable) -> Callable:
    """
    Decorator to convert a blocking I/O function to async.
    
    Usage:
        @async_io
        def read_large_file(path):
            with open(path) as f:
                return f.read()
        
        # Now async
        content = await read_large_file("/path/to/file")
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        manager = AsyncIOManager()
        return await manager.run_in_thread(func, *args, **kwargs)
    
    return wrapper


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_io_manager: Optional[AsyncIOManager] = None


def get_io_manager() -> AsyncIOManager:
    """Get or create global I/O manager"""
    global _io_manager
    if _io_manager is None:
        _io_manager = AsyncIOManager()
    return _io_manager


# Common async file operations
async def async_read_text(path: Union[str, Path], encoding: str = "utf-8") -> str:
    """Read text file asynchronously"""
    return await get_io_manager().read_text(path, encoding)


async def async_write_text(path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
    """Write text file asynchronously"""
    return await get_io_manager().write_text(path, content, encoding)


async def async_read_bytes(path: Union[str, Path]) -> bytes:
    """Read binary file asynchronously"""
    return await get_io_manager().read_bytes(path)


async def async_write_bytes(path: Union[str, Path], content: bytes) -> None:
    """Write binary file asynchronously"""
    return await get_io_manager().write_bytes(path, content)


async def async_read_json(path: Union[str, Path]) -> Any:
    """Read JSON file asynchronously"""
    return await get_io_manager().read_json(path)


async def async_write_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Write JSON file asynchronously"""
    return await get_io_manager().write_json(path, data, indent)


async def async_file_exists(path: Union[str, Path]) -> bool:
    """Check if file exists asynchronously"""
    return await get_io_manager().file_exists(path)


# ============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# ============================================================================

import os


async def safe_open_and_read(path: Union[str, Path], mode: str = "r") -> Union[str, bytes]:
    """
    Safely open and read a file asynchronously.
    
    Replaces common pattern:
        with open(path) as f:
            return f.read()
    """
    if "b" in mode:
        return await async_read_bytes(path)
    else:
        encoding = "utf-8" if "t" in mode or mode == "r" else None
        return await async_read_text(path, encoding or "utf-8")


async def safe_open_and_write(path: Union[str, Path], 
                              content: Union[str, bytes],
                              mode: str = "w") -> None:
    """
    Safely open and write a file asynchronously.
    """
    if "b" in mode:
        await async_write_bytes(path, content)
    else:
        await async_write_text(path, content)


# ============================================================================
# CONTEXT MANAGER FOR TEMP FILES
# ============================================================================

import tempfile
import shutil
from contextlib import asynccontextmanager


@asynccontextmanager
async def async_temp_file(suffix: str = "", prefix: str = "brick_", 
                          delete: bool = True):
    """
    Async context manager for temporary files.
    
    Usage:
        async with async_temp_file(suffix=".json") as tmp_path:
            await async_write_json(tmp_path, data)
            # File automatically cleaned up
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    
    try:
        yield Path(path)
    finally:
        if delete:
            try:
                await async_file_delete(path)
            except:
                pass


@asynccontextmanager
async def async_temp_directory(prefix: str = "brick_", delete: bool = True):
    """
    Async context manager for temporary directories.
    """
    path = tempfile.mkdtemp(prefix=prefix)
    
    try:
        yield Path(path)
    finally:
        if delete:
            def _cleanup():
                shutil.rmtree(path, ignore_errors=True)
            await get_io_manager().run_in_thread(_cleanup)


async def async_file_delete(path: Union[str, Path]) -> bool:
    """Delete file asynchronously"""
    return await get_io_manager().delete_file(path)
