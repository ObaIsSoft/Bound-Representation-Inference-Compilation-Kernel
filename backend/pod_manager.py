import os
import json
import logging
from typing import List, Dict, Any, Optional
from core.hierarchical_resolver import ModularISA

logger = logging.getLogger(__name__)

class PodManager:
    """
    Manages the 'Linked Files' architecture for Modular Pods.
    Discovers assemblies in the file system and maps them to the ISA hierarchy.
    """
    
    def __init__(self, project_root_path: str):
        self.project_root_path = project_root_path

    def discover_and_hydrate(self, pod: ModularISA, folder_relative_path: str) -> bool:
        """
        Scans a folder and populates a ModularISA with linked files and assembly logic.
        """
        pod.source_folder = folder_relative_path
        abs_path = os.path.join(self.project_root_path, folder_relative_path)
        if not os.path.exists(abs_path) or not os.path.isdir(abs_path):
            logger.warning(f"PodManager: Path {abs_path} is not a valid directory.")
            return False

        # 1. Look for pod.json manifest
        manifest_path = os.path.join(abs_path, "pod.json")
        if os.path.exists(manifest_path):
            return self._hydrate_from_manifest(pod, manifest_path)
        
        # 2. Fallback: Auto-linking (All .kcl files)
        return self._hydrate_via_autolink(pod, abs_path, folder_relative_path)

    def _hydrate_from_manifest(self, pod: ModularISA, manifest_path: str) -> bool:
        """Hydrates a pod using a pod.json manifest."""
        try:
            with open(manifest_path, 'r') as f:
                data = json.load(f)
            
            pod.name = data.get("name", pod.name)
            pod.is_merged = data.get("is_merged", False)
            pod.is_folder_linked = True
            pod.assembly_pattern = data.get("assembly_pattern", "MANUAL")
            pod.pattern_params = data.get("pattern_params", {})
            
            components = []
            for file_entry in data.get("files", []):
                components.append({
                    "id": file_entry.get("id"),
                    "path": file_entry.get("path"),
                    "type": file_entry.get("type", "box"), # Phase 24: Geometry Hint
                    "active": file_entry.get("active", True), # Phase 23: Subset selection
                    "transform": file_entry.get("transform", {"translate": [0,0,0], "rotate": [0,0,0]})
                })
            
            pod.linked_components = components
            logger.info(f"PodManager: Hydrated {pod.name} (Merged={pod.is_merged}) from manifest.")
            return True
        except Exception as e:
            logger.error(f"PodManager: Failed to parse manifest at {manifest_path}: {e}")
            return False

    def _hydrate_via_autolink(self, pod: ModularISA, abs_path: str, rel_path: str) -> bool:
        """Automatically links all KCL files in a folder."""
        kcl_files = [f for f in os.listdir(abs_path) if f.endswith(".kcl")]
        
        if not kcl_files:
            return False
            
        pod.is_folder_linked = True
        components = []
        for f in kcl_files:
            components.append({
                "id": f.split(".")[0],
                "path": f,
                "active": True,
                "transform": {"translate": [0,0,0], "rotate": [0,0,0]}
            })
        
        pod.linked_components = components
        logger.info(f"PodManager: Auto-hydrated {pod.name} via directory scan ({len(components)} KCL files).")
        return True

    def merge_pod(self, pod: ModularISA) -> bool:
        """
        Phase 23: Consolidates linked parts into a rigid assembly.
        Triggers snapping and writes the official pod.json manifest.
        """
        if not pod.is_folder_linked:
            return False
            
        logger.info(f"PodManager: Merging assembly '{pod.name}'...")
        
        # 1. Calculate Snapping
        self.calculate_snapping_offsets(pod)
        
        # 2. Update State
        pod.is_merged = True
        
        # 3. Aggregate Metrics & Cache in Constraints
        metrics = self.get_aggregate_metrics(pod)
        pod.constraints["assembly_mass"] = metrics["mass"]
        pod.constraints["assembly_cost"] = metrics["cost"]
        pod.constraints["assembly_power"] = metrics.get("power", 0.0)
        
        # 4. Persistence
        success = self._save_manifest(pod)
        return success

    def unmerge_pod(self, pod: ModularISA):
        """Phase 23: Reverts assembly to independent files."""
        pod.is_merged = False
        pod.constraints.pop("assembly_mass", None)
        pod.constraints.pop("assembly_cost", None)
        pod.constraints.pop("assembly_power", None)
        logger.info(f"PodManager: Unmerged {pod.name}.")

    def calculate_snapping_offsets(self, pod: ModularISA):
        """
        De-hardcoded Snapping Engine (Phase 24).
        Uses assembly_pattern and pattern_params for generic alignment.
        """
        active_comps = [c for c in pod.linked_components if c.get("active", True)]
        if len(active_comps) <= 1: return
        
        pattern = pod.assembly_pattern
        params = pod.pattern_params
        
        logger.info(f"PodManager: Calculating {pattern} snapping for {len(active_comps)} components.")
        
        # Reference component (usually the first one)
        ref = active_comps[0]
        
        if pattern == "RADIAL":
            # Snapping logic: Pivot components around the reference center
            radius = params.get("radius", 0.0)
            axis = params.get("axis", [0, 0, 1]) # Default Z-axis
            
            # Filter out the reference itself from the rotating set if it's the 'hub'
            # (Convention: First active component is the anchor)
            rotators = active_comps[1:]
            count = len(rotators)
            if count == 0: return

            for i, comp in enumerate(rotators):
                angle = (360.0 / count) * i
                # Set rotation based on axis (simplified for Z-axis primarily)
                if axis == [0, 0, 1]:
                    comp["transform"]["rotate"] = [0, 0, angle]
                    comp["transform"]["translate"] = [0, 0, 0] # Assume radius is handled in part geometry or offset
                
        elif pattern == "LINEAR":
            spacing = params.get("spacing", 0.1) # 100mm default
            axis = params.get("axis", [1, 0, 0]) # Default X-axis
            
            for i, comp in enumerate(active_comps):
                offset = [axis[0] * i * spacing, axis[1] * i * spacing, axis[2] * i * spacing]
                comp["transform"]["translate"] = offset
                comp["transform"]["rotate"] = [0, 0, 0]

        elif pattern == "STACK":
            spacing = params.get("spacing", 0.05) # 50mm default
            axis = params.get("axis", [0, 0, 1]) # Default Z-axis
            
            for i, comp in enumerate(active_comps):
                offset = [axis[0] * i * spacing, axis[1] * i * spacing, axis[2] * i * spacing]
                comp["transform"]["translate"] = offset
        
    def _save_manifest(self, pod: ModularISA) -> bool:
        """Writes current assembly state to pod.json."""
        if not pod.source_folder:
            logger.error("PodManager: Cannot save manifest, source_folder is missing.")
            return False

        manifest = {
            "name": pod.name,
            "is_merged": pod.is_merged,
            "assembly_pattern": pod.assembly_pattern,
            "pattern_params": pod.pattern_params,
            "files": pod.linked_components,
            "metadata": {
                "version": "1.1.0",
                "is_auto_generated": True
            }
        }
        
        path = os.path.join(self.project_root_path, pod.source_folder, "pod.json")
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"PodManager: Saved manifest for {pod.name} to {path}")
            return True
        except Exception as e:
            logger.error(f"PodManager: Failed to save manifest: {e}")
            return False

    def get_aggregate_metrics(self, pod: ModularISA) -> Dict[str, float]:
        """
        Aggregates metrics from all active linked components.
        """
        active_comps = [c for c in pod.linked_components if c.get("active", True)]
        return {
            "mass": sum(pod.constraints.get(f"{c['id']}_mass", 0.1) for c in active_comps),
            "cost": sum(pod.constraints.get(f"{c['id']}_cost", 10.0) for c in active_comps),
            "power": sum(pod.constraints.get(f"{c['id']}_power", 0.0) for c in active_comps)
        }
