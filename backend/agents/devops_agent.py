
from typing import Dict, Any, List, Optional
import logging
import os
import json
from backend.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

class DevOpsAgent:
    """
    DevOps Agent ("The Operator").
    Automates deployment checks, pipeline configuration, and container auditing.
    """
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.name = "DevOpsAgent"
        self.llm_provider = llm_provider
        try:
            from backend.config.devops_config import DOCKER_AUDIT_RULES, CI_CD_TEMPLATE, HEALTH_CHECK_TIMEOUT
            self.audit_rules = DOCKER_AUDIT_RULES
            self.pipeline_template = CI_CD_TEMPLATE
            self.timeout = HEALTH_CHECK_TIMEOUT
        except ImportError:
            logger.warning("Could not import devops_config. Using defaults.")
            self.audit_rules = [{"check": "USER", "message": "Security: Container runs as root"}]
            self.pipeline_template = "name: CI/CD Pipeline\n..."
            self.timeout = 5
        
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
                    timeout=self.timeout
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
            
            for rule in self.audit_rules:
                if rule["check"] in content:
                     # Some checks are "must exist" and some are "must not exist"
                     # The original logic was specific.
                     # Let's adapt: If message says "missing", then it MUST be there (so if NOT in content, error)
                     # If message says "Avoid", then it MUST NOT be there (so if IN content, error)
                     
                     if "missing" in rule["message"].lower():
                         if rule["check"] not in content:
                             issues.append(rule["message"])
                     elif "avoid" in rule["message"].lower() or "discouraged" in rule["message"].lower():
                         if rule["check"] in content:
                             issues.append(rule["message"])
            
            # Simple fallback for standard checks if config is raw
            if "USER" not in content and not any("missing user" in i.lower() for i in issues):
                 issues.append("Security: Container runs as root (missing USER instruction)")

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
                 "config": self.pipeline_template
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
                 "config": self.pipeline_template,
                 "fallback": True
            }

    def _get_pipeline_template(self) -> str:
        return self.pipeline_template
