from typing import Dict, Any, List
import subprocess
import os
import logging

logger = logging.getLogger(__name__)

class ShellAgent:
    """
    Shell Agent - CLI Execution.
    
    Executes system commands with:
    - Working directory persistence
    - Environment variable access
    - Output capture
    - Error handling
    
    NOTE: This agent provides direct system access - use with caution.
    """
    
    def __init__(self):
        self.name = "ShellAgent"
        self.cwd = os.getcwd()
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute shell command.
        
        Args:
            params: {
                "cmd": str,  # Command to execute
                "args": List[str],  # Optional arguments
                "cwd": Optional str  # Working directory
            }
        
        Returns:
            {
                "stdout": str,
                "stderr": str,
                "returncode": int,
                "logs": List of operation logs
            }
        """
        cmd = params.get("cmd", "")
        args = params.get("args", [])
        cwd = params.get("cwd", self.cwd)
        
        logs = [f"[SHELL] Executing: {cmd} {' '.join(args)}", f"[SHELL] Working directory: {cwd}"]
        
        if not cmd:
            return {
                "stdout": "",
                "stderr": "No command provided",
                "returncode": 1,
                "logs": logs + ["[SHELL] ✗ No command"]
            }
        
        # Build full command
        full_cmd = f"{cmd} {' '.join(args)}".strip()
        
        try:
            # Handle cd separately (stateful)
            if cmd == "cd":
                target = args[0] if args else os.path.expanduser("~")
                new_path = os.path.abspath(os.path.join(cwd, target))
                
                if os.path.isdir(new_path):
                    self.cwd = new_path
                    logs.append(f"[SHELL] ✓ Changed directory to {new_path}")
                    return {
                        "stdout": f"cd {new_path}",
                        "stderr": "",
                        "returncode": 0,
                        "logs": logs
                    }
                else:
                    return {
                        "stdout": "",
                        "stderr": f"cd: no such file or directory: {target}",
                        "returncode": 1,
                        "logs": logs + [f"[SHELL] ✗ Invalid directory"]
                    }
            
            # Execute command
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=os.environ.copy()
            )
            
            logs.append(f"[SHELL] Exit code: {result.returncode}")
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "logs": logs
            }
            
        except Exception as e:
            logs.append(f"[SHELL] ✗ Exception: {str(e)}")
            return {
                "stdout": "",
                "stderr": f"Shell Error: {str(e)}",
                "returncode": 1,
                "logs": logs
            }
