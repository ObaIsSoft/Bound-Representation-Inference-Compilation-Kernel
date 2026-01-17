import unittest
import sys
import os
import shutil
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.component_agent import ComponentAgent
# Mock Supabase before importing
sys.modules['database.supabase_client'] = MagicMock()

class TestComponentInstall(unittest.TestCase):

    def setUp(self):
        self.agent = ComponentAgent()
        self.test_dir = "tests/temp_data"
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('utils.mesh_to_sdf_bridge.MeshSDFBridge.bake_sku_to_sdf')
    def test_install_component_mocked(self, mock_bake):
        """Test installation logic with mocked bridge"""
        
        # Setup mock return
        mock_bake.return_value = (
            None, # sdf_grid (not needed for logic test)
            {"resolution": 32, "bounds": [[0,0,0],[1,1,1]]} # metadata
        )
        
        # Run install
        result = self.agent.install_component("test_component_123", mesh_path=None)
        
        # Verify
        self.assertEqual(result["status"], "installed")
        self.assertEqual(result["component_id"], "test_component_123")
        self.assertEqual(result["sdf_metadata"]["resolution"], 32)
        
        # Verify bridge was called (path resolved to synthetic or default)
        mock_bake.assert_called_once()
        print("\n✓ PASS: Component installation logic verified")

if __name__ == '__main__':
    unittest.main()
