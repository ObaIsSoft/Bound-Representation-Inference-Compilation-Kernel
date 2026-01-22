from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class NexusAgent:
    """
    Nexus Agent - Context Navigation.
    
    Manages:
    - Context tree navigation (filesystem-like access to design).
    - Querying relationships between components.
    - Knowledge graph traversal.
    """
    
    def __init__(self):
        self.name = "NexusAgent"
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Navigate or query context (Real File System).
        """
        operation = params.get("operation", "list")
        path = params.get("path", "/")
        root_dir = "projects" # Corresponds to ProjectManager storage
        
        # Ensure project dir exists
        import os
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        logs = [f"[NEXUS] Operation: {operation} at {path}"]
        data = None
        
        if operation == "list":
            # List .brick files in the directory
            try:
                files = [f for f in os.listdir(root_dir) if f.endswith(".brick")]
                # Format for Frontend (id, name, type)
                data = []
                for f in files:
                    data.append({
                        "id": f, 
                        "name": f, 
                        "type": "file", 
                        "parentId": None
                    })
                
                # Add a dummy folder for Structure (if needed, or just flat list for now)
                data.append({"id": "legacy", "name": "Legacy Projects", "type": "folder", "parentId": None})
                
                logs.append(f"Found {len(files)} projects.")
            except Exception as e:
                logs.append(f"Error listing files: {str(e)}")
                data = []
            
        elif operation == "read":
            # Read content of a specific .brick file
            target_file = os.path.join(root_dir, path.strip("/"))
            if os.path.exists(target_file):
                import json
                try:
                    with open(target_file, 'r') as f:
                        content = json.load(f)
                    data = content
                    logs.append("File loaded successfully.")
                except Exception as e:
                    logs.append(f"Error reading file: {str(e)}")
            else:
                logs.append("File not found.")
            
        return {
            "data": data,
            "current_path": path,
            "logs": logs
        }
