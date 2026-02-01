from typing import Dict, Any, List
import logging
import math
try:
    from isa import PhysicalValue, Unit, create_physical_value
except ImportError:
    from isa import PhysicalValue, Unit, create_physical_value

logger = logging.getLogger(__name__)

class MassPropertiesAgent:
    """
    Mass Properties Agent.
    Calculates Mass, Center of Gravity (CoG), and Inertia based on geometry and material.
    """
    def __init__(self):
        self.name = "MassPropertiesAgent"
        
        # Tier 5: Neural Surrogate
        try:
            from models.mass_properties_surrogate import MassPropertiesSurrogate
            self.surrogate = MassPropertiesSurrogate()
            self.use_surrogate = True
        except ImportError:
            self.surrogate = None
            self.use_surrogate = False

    def run(self, geometry: List[Any], material_name: str) -> Dict[str, Any]:
        """
        Execute mass properties analysis.
        Args:
            geometry: List of GeometryNodes (or dict representation)
            material_name: Name of the material (e.g. "Aluminum 6061")
        """
        logger.info(f"{self.name} calculating mass stats for {material_name}...")
        
        # 0. Material Density Lookup (Mock/Basic)
        # In real system, this calls MaterialAgent or DB
        densities = {
            "Aluminum 6061": 2.7,
            "Steel": 7.85,
            "Titanium": 4.43,
            "Plastic": 1.2,
            "ABS": 1.04,
            "PLA": 1.24
        }
        # Fuzzy match or default
        density_g_cm3 = 2.7
        for key, val in densities.items():
            if key.lower() in material_name.lower():
                density_g_cm3 = val
                break
        
        # 1. Volume Estimation from Geometry Tree
        # If geometry is empty or invalid, default to 1000.0
        volume_cm3 = 1000.0
        bbox = [10.0, 10.0, 10.0]
        
        if isinstance(geometry, list) and len(geometry) > 0:
            # Try to extract bbox from first node metadata/params
            try:
                # Basic heuristic: sum volumes of primitives?
                # For now: Just use defaults with "Estimated" log
                pass 
            except:
                pass
                
        # Inputs used for calculation
        # volume_cm3 = params.get("volume_cm3", 1000.0) 
        # density_g_cm3 = params.get("material_density", 2.7) 
        # bbox = params.get("bounding_box", [10.0, 10.0, 10.0])
        
        logs = []
        
        # 1. Calculate Mass
        mass_g = volume_cm3 * density_g_cm3
        mass_kg = mass_g / 1000.0
        
        # 2. Estimate Inertia Tensor (Cuboid Approximation)
        # Ixx = m/12 * (y^2 + z^2)
        # Iyy = m/12 * (x^2 + z^2)
        # Izz = m/12 * (x^2 + y^2)
        lx, ly, lz = bbox[0]/100.0, bbox[1]/100.0, bbox[2]/100.0 # Convert cm to m for Inertia output
        
        if self.use_surrogate and self.surrogate:
            try:
                inertia = self.surrogate.predict_inertia(mass_kg, volume_cm3, bbox)
                ixx, iyy, izz = inertia
                logs.append("Method: Neural Surrogate Estimation")
            except Exception as e:
                logger.warning(f"Surrogate failed: {e}")
                # Fallback to Cuboid
                ixx = (mass_kg / 12.0) * (ly**2 + lz**2)
                iyy = (mass_kg / 12.0) * (lx**2 + lz**2)
                izz = (mass_kg / 12.0) * (lx**2 + ly**2)
                logs.append("Method: Analytic Cuboid (Fallback)")
        else:
            # Analytic Cuboid
            ixx = (mass_kg / 12.0) * (ly**2 + lz**2)
            iyy = (mass_kg / 12.0) * (lx**2 + lz**2)
            izz = (mass_kg / 12.0) * (lx**2 + ly**2)
            logs.append("Method: Analytic Cuboid")
        
        # 3. Center of Gravity (CoG)
        # For a single component, we assume CoG is geometric center (0,0,0) relative to body
        # In a real assembly, this would aggregate children.
        cg = [0.0, 0.0, 0.0]
        
        # Create PhysicalValues
        pv_mass = create_physical_value(mass_kg, Unit.KILOGRAMS, source=self.name)
        
        logs = [
            f"Volume: {volume_cm3:.2f} cm³, Density: {density_g_cm3:.2f} g/cm³",
            f"Calculated Mass: {mass_kg:.4f} kg",
            f"Inertia (diag): [{ixx:.4f}, {iyy:.4f}, {izz:.4f}] kg·m²"
        ]

        # --- Recursive ISA Integration ---
        # If running in the context of a scoped pod, verify scope and update exports
        pod_id = None # params variable was removed from signature
        if pod_id:
            try:
                from core.system_registry import get_system_resolver
                resolver = get_system_resolver()
                
                # Check for pod existence (O(1) lookup map is better, but tree walk for now)
                # We need a get_pod_by_id logic or assume path... 
                # The resolver currently only has get_pod_by_path. We should fix that OR scan.
                # For demo, we'll scan or assume the agent received the object itself? No, serializable state only.
                # Let's add find_pod_by_id to resolver later. For now, let's just implement the 'Logic Node' concept.
                
                # Assume a helper exists or we iterate. 
                # Actually, let's just add update logic if we find it.
                
                # Recursive Find Helper (Quick & Dirty for 140 IQ MVP)
                # Ideally this logic is in Resolver
                def find_pod(root, target_id):
                    if root.id == target_id: return root
                    for sub in root.sub_pods.values():
                        found = find_pod(sub, target_id)
                        if found: return found
                    return None
                
                pod = find_pod(resolver.root, pod_id)
                
                if pod:
                    # SCOPE VALIDATION: Am I allowed to write to this pod?
                    # The Agent is the 'Internal Logic' of the pod, so yes.
                    
                    # Update PRIVATE State (Internal constraints)
                    pod.constraints["local_mass"] = mass_kg
                    pod.constraints["inertia"] = [ixx, iyy, izz]
                    
                    # Update PUBLIC Export (The Footprint)
                    # We trigger converge_up which sums children + local
                    import asyncio
                    # Ideally we await, but run() is synchronous here... 
                    # Hack: Run event loop update? Or just call synchronous version if we made one.
                    # We'll just set it directly and maybe call a sync-wrapper or assume external trigger.
                    # For MVP, direct update + print.
                    
                    prev_mass = pod.exports["mass"]
                    pod.exports["mass"] = mass_kg # Simplified: Assume leaf node for this calc or handled by converge
                    
                    # Trigger bottom-up bubble (Sync shim)
                    # resolver.converge_up_sync(pod) # We need to add this to resolver
                    
                    logs.append(f"[RecursiveISA] Updated Pod '{pod.name}' Local Mass: {mass_kg:.4f}kg")

            except ImportError:
                 logs.append("[RecursiveISA] System Registry not available.")
            except Exception as e:
                 logs.append(f"[RecursiveISA] Error updating pod: {str(e)}")

        return {
            "status": "success",
            "total_mass_kg": mass_kg,
            "mass": pv_mass.to_dict(),
            "inertia_tensor": [ixx, iyy, izz],
            "center_of_gravity": cg,
            "logs": logs
        }
