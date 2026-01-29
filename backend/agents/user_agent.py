import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class UserAgent:
    """
    Manages user profile data for Single User Mode.
    Persists data to a local JSON file.
    """
    
    def __init__(self):
        # Determine path relative to this file
        # this file is in backend/agents/
        # data is in backend/data/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(base_dir, "data")
        self.profile_path = os.path.join(self.data_dir, "user_profile.json")
        
        # Ensure data dir exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Default Profile
        self.default_profile = {
            "name": "Obafemi",
            "email": "user@example.com",
            "role": "BRICK Architect",
            "avatar_initials": "OB",
            "plan": "Pro License",
            "theme_preference": "dark",
            "created_at": "2026-01-01T00:00:00Z"
        }

    def _load_data(self) -> Dict[str, Any]:
        """Loads profile from disk or creates default."""
        if not os.path.exists(self.profile_path):
            self._save_data(self.default_profile)
            return self.default_profile
            
        try:
            with open(self.profile_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load user profile: {e}")
            return self.default_profile

    def _save_data(self, data: Dict[str, Any]):
        """Saves profile to disk."""
        try:
            with open(self.profile_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user profile: {e}")
            raise e

    def get_profile(self) -> Dict[str, Any]:
        """Returns the current user profile."""
        return self._load_data()

    def update_profile(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates the user profile with provided fields.
        Returns the updated profile.
        """
        current = self._load_data()
        
        # Merge updates (shallow merge is usually sufficient for this flat structure)
        # Prevent overwriting protected fields if any (simulate security)
        protected_fields = ["plan", "created_at"] 
        
        for k, v in updates.items():
            if k not in protected_fields:
                current[k] = v
                
        # Auto-update initials if name changes
        if "name" in updates:
            name_parts = updates["name"].split()
            if len(name_parts) >= 2:
                current["avatar_initials"] = (name_parts[0][0] + name_parts[1][0]).upper()
            elif len(name_parts) == 1:
                 current["avatar_initials"] = name_parts[0][:2].upper()
        
        self._save_data(current)
        return current
