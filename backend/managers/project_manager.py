
import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Any

class ProjectManager:
    def __init__(self, storage_dir: str = "projects"):
        self.storage_dir = storage_dir
        # Ensure 'main' branch exists by default
        os.makedirs(os.path.join(self.storage_dir, "main"), exist_ok=True)
        print(f"[ProjectManager] Initialized with storage: {os.path.abspath(self.storage_dir)}")

    def _get_branch_path(self, branch: str) -> str:
        path = os.path.join(self.storage_dir, branch)
        os.makedirs(path, exist_ok=True)  # Auto-create on access
        return path

    def save_project(self, project_data: Dict[str, Any], filename: str = "save.brick", branch: str = "main") -> str:
        """
        Saves the project state to a JSON file in the specified branch.
        Auto-creates branch folder if missing.
        """
        # Ensure manifest exists
        if "manifest" not in project_data:
            project_data["manifest"] = {}
        
        project_data["manifest"]["last_saved"] = datetime.now().isoformat()
        project_data["manifest"]["version"] = "1.0.0"
        project_data["manifest"]["branch"] = branch

        branch_path = self._get_branch_path(branch)  # Auto-creates folder
        filepath = os.path.join(branch_path, filename)
        
        # Backup existing if present
        if os.path.exists(filepath):
            backup_path = filepath + ".bak"
            shutil.copy2(filepath, backup_path)

        with open(filepath, "w") as f:
            json.dump(project_data, f, indent=2)
        
        print(f"[ProjectManager] Saved to: {filepath}")
        return os.path.abspath(filepath)

    def load_project(self, filename: str = "save.brick", branch: str = "main") -> Dict[str, Any]:
        """Loads project state from a JSON file in the specified branch."""
        filepath = os.path.join(self._get_branch_path(branch), filename)
        
        if not os.path.exists(filepath):
            # Fallback to searching in root or 'main' if specific branch fails (optional, but good for UX)
            fallback_path = os.path.join(self._get_branch_path("main"), filename)
            if os.path.exists(fallback_path):
                 with open(fallback_path, "r") as f: return json.load(f)
            raise FileNotFoundError(f"Project file not found: {filepath}")
            
        with open(filepath, "r") as f:
            return json.load(f)

    def delete_project(self, filename: str = "save.brick", branch: str = "main") -> bool:
        """Deletes a project file from the specified branch."""
        filepath = os.path.join(self._get_branch_path(branch), filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"[ProjectManager] Deleted: {filepath}")
            return True
        return False

    def list_projects(self, branch: str = "main"):
        """Returns list of .brick files in storage (legacy flat support handled by defaulting to main)."""
        path = self._get_branch_path(branch)
        if not os.path.exists(path): return []
        return [f for f in os.listdir(path) if f.endswith(".brick")]

    def get_history(self, branch: str = "main") -> List[Dict[str, Any]]:
        """
        Returns history for the active branch.
        """
        commits = []
        branch_path = self._get_branch_path(branch)
        
        try:
            if not os.path.exists(branch_path):
                return []
                
            files = [f for f in os.listdir(branch_path) if f.endswith(".brick")]
            for f in files:
                path = os.path.join(branch_path, f)
                stats = os.stat(path)
                mtime = datetime.fromtimestamp(stats.st_mtime)
                
                commits.append({
                    "hash": f"{hash(f + str(stats.st_mtime)) & 0xffffff:06x}",
                    "message": f"Snapshot: {f}",
                    "author": "USER",
                    "time": mtime.strftime("%H:%M:%S"),
                    "timestamp": mtime.isoformat(),
                    "branch": branch
                })
        except Exception as e:
            print(f"Error reading history: {e}")
            
        commits.sort(key=lambda x: x["timestamp"], reverse=True)
        return commits

    # --- Real Branching Support ---
    def get_branches(self):
        """Scans subdirectories to find branches."""
        branches = []
        try:
            # List all directories in storage_dir
            for item in os.listdir(self.storage_dir):
                item_path = os.path.join(self.storage_dir, item)
                if os.path.isdir(item_path):
                    # Count commits
                    commits = len([f for f in os.listdir(item_path) if f.endswith(".brick")])
                    branches.append({
                        "name": item, 
                        "active": (item == "main"), # Logic to rely on request param later
                        "commits": commits
                    })
        except Exception as e:
            print(f"Error listing branches: {e}")
            
        if not branches:
             branches.append({"name": "main", "active": True, "commits": 0})
             
        return branches

    def create_branch(self, name: str, source_branch: str = "main"):
        """Creates a new branch (folder) and copies the latest state from source."""
        new_path = self._get_branch_path(name)
        if os.path.exists(new_path):
            raise ValueError(f"Branch '{name}' already exists.")
            
        os.makedirs(new_path)
        
        # Copy latest file from source to dest to initialize
        history = self.get_history(source_branch)
        if history:
            latest_file = history[0]["message"].replace("Snapshot: ", "")
            src_file_path = os.path.join(self._get_branch_path(source_branch), latest_file)
            if os.path.exists(src_file_path):
                 shutil.copy2(src_file_path, os.path.join(new_path, latest_file))
        
        return {"name": name, "commits": 1 if history else 0}

    def create_commit(self, message: str, project_data: Dict[str, Any], branch: str = "main"):
        """Creates a timestamped snapshot as a commit."""
        timestamp = int(time.time())
        filename = f"commit_{timestamp}.brick"
        path = self.save_project(project_data, filename, branch=branch)
        return {
            "hash": f"{timestamp & 0xffffff:06x}",
            "message": message,
            "path": path,
            "branch": branch
        }

