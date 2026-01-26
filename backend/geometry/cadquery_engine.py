import logging
import json
import subprocess
import os
import sys
from .base_engine import BaseGeometryEngine, GeometryRequest, GeometryResult

logger = logging.getLogger(__name__)

class CadQueryEngine(BaseGeometryEngine):
    """
    COLD PATH ENGINE
    Uses CadQuery (via OCCT) for high-fidelity export (STEP/IGES).
    Runs in a separate process via `_worker_cadquery.py` to ensure crash isolation.
    """
    
    def build(self, request: GeometryRequest) -> GeometryResult:
        try:
            # Prepare payload
            payload = {
                "request_id": request.request_id,
                "tree": request.tree,
                "output_format": request.output_format
            }
            input_json = json.dumps(payload)
            
            # Locate worker script
            worker_path = os.path.join(os.path.dirname(__file__), "_worker_cadquery.py")
            
            # Execute Worker
            # Use same python interpreter
            cmd = [sys.executable, worker_path]
            
            logger.info(f"Invoking Cold Path Worker for {request.request_id}...")
            
            process = subprocess.run(
                cmd,
                input=input_json,
                capture_output=True,
                text=True,
                timeout=60 # 60s max for export
            )
            
            if process.returncode != 0:
                logger.error(f"Usage Worker StdErr: {process.stderr}")
                return GeometryResult(success=False, error=f"Worker Failed: {process.stderr}")
            
            # Parse output
            try:
                # Stdout should contain the JSON response
                # Note: If worker printed other stuff to stdout, this breaks. 
                # Worker is configured to log to stderr.
                output_data = json.loads(process.stdout)
                
                if output_data.get("success"):
                     return GeometryResult(
                         success=True,
                         file_path=output_data.get("file_path")
                     )
                else:
                     return GeometryResult(success=False, error=output_data.get("error"))
                     
            except json.JSONDecodeError:
                return GeometryResult(success=False, error=f"Invalid JSON from worker: {process.stdout}")
                
        except subprocess.TimeoutExpired:
            return GeometryResult(success=False, error="Cold Path Timeout (Operation took too long)")
        except Exception as e:
            logger.error(f"CadQuery Engine Error: {e}")
            return GeometryResult(success=False, error=str(e))
