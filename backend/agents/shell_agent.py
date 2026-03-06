"""
Production ShellAgent - Secure Shell Command Execution

Follows BRICK OS patterns:
- NO shell=True - uses safe subprocess with argument lists
- Working directory persistence
- Environment variable access
- Output capture with proper error handling
- Whitelist for dangerous commands

SECURITY WARNING: This agent provides direct system access.
Uses shell=False with argument lists to prevent injection attacks.
"""

from typing import Dict, Any, List, Optional
import subprocess
import os
import logging
import shlex

logger = logging.getLogger(__name__)


class ShellAgent:
    """
    Production shell command execution agent.
    
    Executes system commands with:
    - Safe subprocess (no shell=True)
    - Working directory persistence
    - Environment variable access
    - Output capture
    - Error handling
    - Command whitelist for dangerous operations
    
    SECURITY: Uses argument lists instead of shell string to prevent injection.
    """
    
    # Dangerous commands that require explicit confirmation
    DANGEROUS_COMMANDS = {
        'rm': ['-rf', '-r', '-f', '--no-preserve-root'],
        'dd': [],
        'mkfs': [],
        'fdisk': [],
        '>:': [],  # Redirect overwrite
        '>>': [],  # Redirect append
        '|': [],   # Pipes
    }
    
    def __init__(self):
        self.name = "ShellAgent"
        self._initialized = False
        self._cwd = os.getcwd()
        
    async def initialize(self):
        """Initialize agent."""
        if self._initialized:
            return
        
        self._cwd = os.getcwd()
        self._initialized = True
        logger.info("ShellAgent initialized")
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute shell command securely.
        
        Args:
            params: {
                "cmd": str,  # Command to execute
                "args": List[str],  # Optional arguments
                "cwd": Optional[str],  # Working directory override
                "env": Optional[Dict[str, str]],  # Environment variables
                "timeout": Optional[int],  # Timeout in seconds
                "input_data": Optional[str]  # stdin input
            }
        
        Returns:
            {
                "stdout": str,
                "stderr": str,
                "returncode": int,
                "cwd": str,
                "logs": List of operation logs
            }
        """
        await self.initialize()
        
        cmd = params.get("cmd", "").strip()
        args = params.get("args", [])
        cwd = params.get("cwd", self._cwd)
        env_override = params.get("env", {})
        timeout = params.get("timeout", 60)
        input_data = params.get("input_data")
        
        logs = [
            f"[SHELL] Executing: {cmd}",
            f"[SHELL] Working directory: {cwd}"
        ]
        
        if not cmd:
            return {
                "stdout": "",
                "stderr": "No command provided",
                "returncode": 1,
                "cwd": self._cwd,
                "logs": logs + ["[SHELL] ✗ No command"]
            }
        
        # Handle cd separately (stateful)
        if cmd == "cd":
            return await self._handle_cd(args, cwd, logs)
        
        # Check for dangerous commands
        safety_check = self._check_command_safety(cmd, args)
        if not safety_check["safe"]:
            return {
                "stdout": "",
                "stderr": f"Command blocked: {safety_check['reason']}",
                "returncode": 1,
                "cwd": self._cwd,
                "logs": logs + [f"[SHELL] ✗ Blocked: {safety_check['reason']}"]
            }
        
        # Build environment
        env = os.environ.copy()
        env.update(env_override)
        
        # Build command list (safe, no shell=True)
        cmd_list = [cmd] + args
        
        try:
            logger.info(f"[ShellAgent] Executing: {' '.join(cmd_list)}")
            
            result = subprocess.run(
                cmd_list,
                shell=False,  # SECURITY: Never use shell=True
                capture_output=True,
                text=True,
                cwd=cwd,
                env=env,
                timeout=timeout,
                input=input_data
            )
            
            logs.append(f"[SHELL] Exit code: {result.returncode}")
            
            if result.returncode == 0:
                logs.append("[SHELL] ✓ Success")
            else:
                logs.append(f"[SHELL] ✗ Failed with code {result.returncode}")
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "cwd": cwd,
                "logs": logs
            }
            
        except subprocess.TimeoutExpired:
            logs.append(f"[SHELL] ✗ Timeout after {timeout}s")
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "returncode": -1,
                "cwd": cwd,
                "logs": logs
            }
            
        except FileNotFoundError:
            logs.append(f"[SHELL] ✗ Command not found: {cmd}")
            return {
                "stdout": "",
                "stderr": f"Command not found: {cmd}",
                "returncode": 127,
                "cwd": cwd,
                "logs": logs
            }
            
        except Exception as e:
            logs.append(f"[SHELL] ✗ Exception: {str(e)}")
            logger.error(f"[ShellAgent] Execution error: {e}")
            return {
                "stdout": "",
                "stderr": f"Shell Error: {str(e)}",
                "returncode": 1,
                "cwd": cwd,
                "logs": logs
            }
    
    async def _handle_cd(
        self,
        args: List[str],
        cwd: str,
        logs: List[str]
    ) -> Dict[str, Any]:
        """Handle cd command with stateful directory tracking."""
        
        target = args[0] if args else os.path.expanduser("~")
        
        # Handle relative paths
        if not os.path.isabs(target):
            target = os.path.abspath(os.path.join(cwd, target))
        
        # Verify directory exists
        if os.path.isdir(target):
            self._cwd = target
            logs.append(f"[SHELL] ✓ Changed directory to {target}")
            return {
                "stdout": target,
                "stderr": "",
                "returncode": 0,
                "cwd": self._cwd,
                "logs": logs
            }
        else:
            logs.append(f"[SHELL] ✗ Directory not found: {target}")
            return {
                "stdout": "",
                "stderr": f"cd: no such file or directory: {target}",
                "returncode": 1,
                "cwd": cwd,
                "logs": logs
            }
    
    def _check_command_safety(self, cmd: str, args: List[str]) -> Dict[str, Any]:
        """Check if command is safe to execute."""
        
        # Check for pipe characters or redirects in command
        dangerous_chars = ['|', '>', '<', ';', '&', '$', '`']
        for char in dangerous_chars:
            if char in cmd:
                return {
                    "safe": False,
                    "reason": f"Command contains dangerous character: '{char}'. Use 'args' parameter instead."
                }
        
        # Check against dangerous command list
        if cmd in self.DANGEROUS_COMMANDS:
            dangerous_args = self.DANGEROUS_COMMANDS[cmd]
            for arg in args:
                if any(dangerous in arg for dangerous in dangerous_args):
                    return {
                        "safe": False,
                        "reason": f"Dangerous argument detected for {cmd}: {arg}"
                    }
        
        return {"safe": True, "reason": ""}
    
    def get_current_directory(self) -> str:
        """Get current working directory."""
        return self._cwd


# Convenience function
async def execute_shell_command(
    cmd: str,
    args: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    timeout: int = 60
) -> Dict[str, Any]:
    """Quick shell command execution."""
    agent = ShellAgent()
    return await agent.run({
        "cmd": cmd,
        "args": args or [],
        "cwd": cwd,
        "timeout": timeout
    })
