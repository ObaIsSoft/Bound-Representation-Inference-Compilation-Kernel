
from typing import Dict, Any, List, Optional
import logging
import os
import json
from llm.provider import LLMProvider

logger = logging.getLogger(__name__)

class DevOpsAgent:
    """
    DevOps Agent ("The Operator").
    Automates deployment checks, pipeline configuration, and container auditing.
    """
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.name = "DevOpsAgent"
        self.llm_provider = llm_provider
        
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute DevOps tasks.
        Supported actions:
        - "health_check": Verify container/system health
        - "generate_pipeline": Create CI/CD config
        - "audit_dockerfile": Check Dockerfile for security
        """
        action = params.get("action", "health_check")
        logger.info(f"{self.name} executing action: {action}")
        
        if action == "health_check":
            return self.check_system_health()
        elif action == "generate_pipeline":
            return self.generate_pipeline_config(params.get("context", {}))
        elif action == "audit_dockerfile":
            return self.audit_dockerfile(params.get("dockerfile_path", "Dockerfile"))
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
            
    def check_system_health(self) -> Dict[str, Any]:
        """Check system resources and container status using REAL Docker CLI."""
        import shutil
        import subprocess
        
        # Disk Usage
        total, used, free = shutil.disk_usage("/")
        
        # Docker Check
        docker_status = "offline"
        containers_running = 0
        services_status = {}
        
        if shutil.which("docker"):
            try:
                # Run 'docker ps' to count containers
                result = subprocess.run(
                    ["docker", "ps", "-q"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    docker_status = "online"
                    container_ids = result.stdout.strip().split('\n')
                    # Filter empty strings if no containers
                    containers_running = len([c for c in container_ids if c])
                    
                    # Check specialized containers if possible (by name)
                    # e.g. docker ps --format "{{.Names}}"
                    names_res = subprocess.run(
                        ["docker", "ps", "--format", "{{.Names}}"],
                        capture_output=True,
                        text=True
                    )
                    running_names = names_res.stdout
                    services_status["database"] = "online" if "db" in running_names or "postgres" in running_names else "offline"
                    services_status["backend"] = "online" if "backend" in running_names else "offline"
                    
            except Exception as e:
                logger.warning(f"Docker check failed: {e}")
                docker_status = "error"
        
        return {
            "status": "healthy" if docker_status == "online" else "degraded",
            "disk_usage_percent": (used / total) * 100,
            "docker_engine": docker_status,
            "containers_running": containers_running,
            "services_detected": services_status
        }

    def audit_dockerfile(self, path: str) -> Dict[str, Any]:
        """Lint Dockerfile for security best practices."""
        issues = []
        if not os.path.exists(path):
            return {"status": "error", "message": "Dockerfile not found"}
            
        try:
            with open(path, 'r') as f:
                content = f.read()
                
            # Rule 1: Root user check
            if "USER" not in content:
                issues.append("Security: Container runs as root (missing USER instruction)")
                
            # Rule 2: Latest tag check
            if ":latest" in content:
                issues.append("Stability: Using ':latest' tag is discouraged")
            
            # Rule 3: Sudo usage
            if "sudo" in content:
                issues.append("Security: Avoid using sudo in Dockerfile")
                
            return {
                "status": "audited",
                "secure": len(issues) == 0,
                "issues": issues
            }
        except Exception as e:
            logger.error(f"Audit failed: {e}")
            return {"status": "error", "message": str(e)}

    def generate_pipeline_config(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CI/CD pipeline configuration (GitHub Actions)."""
        if not self.llm_provider:
             # Template Fallback
             return {
                 "status": "success",
                 "config": self._get_pipeline_template()
             }
             
        prompt = f"""
        Generate a GitHub Actions CI/CD configuration (YAML) for this project.
        Context: {context}
        Include steps for: Linting, Testing, Building Docker Image.
        """
        
        try:
            config = self.llm_provider.generate(prompt)
            return {
                "status": "success",
                "config": config,
                "source": "LLM"
            }
        except Exception as e:
            return {
                 "status": "success",
                 "config": self._get_pipeline_template(),
                 "fallback": True
            }

    def _get_pipeline_template(self) -> str:
        return """
name: CI/CD Pipeline
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Test
      run: python -m pytest
"""
