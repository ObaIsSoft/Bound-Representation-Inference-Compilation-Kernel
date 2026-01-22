
import os
import json
import shutil
import unittest
from managers.project_manager import ProjectManager

class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_projects"
        self.pm = ProjectManager(storage_dir=self.test_dir)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_and_load(self):
        # 1. Create Mock Data
        mock_data = {
            "geometry": {
                "id": "root",
                "name": "Test Root",
                "children": []
            },
            "physics": {
                "velocity": 100,
                "mass": 50
            },
            "sketch": [
                {"x": 10, "y": 10},
                {"x": 20, "y": 20}
            ]
        }
        
        # 2. Save
        path = self.pm.save_project(mock_data, "test_save.brick")
        self.assertTrue(os.path.exists(path))
        
        # 3. Load
        loaded_data = self.pm.load_project("test_save.brick")
        
        # 4. Verify
        self.assertEqual(loaded_data["geometry"]["name"], "Test Root")
        self.assertEqual(loaded_data["physics"]["velocity"], 100)
        self.assertEqual(len(loaded_data["sketch"]), 2)
        
        # 5. Verify Metadata Injection
        self.assertIn("manifest", loaded_data)
        self.assertIn("version", loaded_data["manifest"])
        
    def test_backup_creation(self):
        mock_data = {"test": 1}
        filename = "backup_test.brick"
        
        # First Save
        self.pm.save_project(mock_data, filename)
        
        # Second Save
        self.pm.save_project(mock_data, filename)
        
        # Verify Backup Exists
        backup_path = os.path.join(self.test_dir, filename + ".bak")
        self.assertTrue(os.path.exists(backup_path))

if __name__ == '__main__':
    unittest.main()
