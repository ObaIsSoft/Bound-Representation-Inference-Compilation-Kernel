from typing import Dict, Any, List
import logging
import re

logger = logging.getLogger(__name__)

class EnvironmentAgent:
    """
    Parses user intent to determine the operating environment.
    Sets gravity, atmospheric pressure, temperature, fluid properties,
    magnetic fields, and solar irradiance.
    """
    def get_manifest(self) -> List[str]:
        """Return list of all supported environment types."""
        return [
            "AERO", "NAVAL", "SPACE", "GROUND", "INDUSTRIAL", "BIO",
            "MOON", "MARS", "VENUS", "TITAN", "EUROPA", 
            "JUPITER_ORBIT", "SATURN_ORBIT", "ASTEROID", "VOLCANO", 
            "HURRICANE", "UNDERSEA", "STATIC"
        ]

    def run(self, user_intent: str) -> Dict[str, Any]:
        """
        Determine environment and vehicle regime from user intent.
        
        Returns:
            dict with type, gravity, fluid_density, pressure, temperature, 
            viscosity, magnetic_field, solar_flux, AND regime.
        """
        intent_lower = user_intent.lower()
        
        # 1. Determine Physical Environment (Location)
        env_data = self._determine_location(intent_lower)
        
        # 2. Determine Vehicle Regime (Mode of Operation)
        regime = self._determine_regime(intent_lower, env_data["type"])
        
        # Merge
        env_data["regime"] = regime
        
        # Physics Enhancement: Radiation Pressure (P = Phi / c)
        if regime == "SPACE" or env_data.get("fluid_density", 0) == 0:
            c = 299792458
            flux = env_data.get("solar_flux", 0)
            # Radiation pressure (perfect absorption)
            env_data["radiation_pressure_pa"] = flux / c
        else:
            env_data["radiation_pressure_pa"] = 0.0
            
        # Physics Enhancement: Magnetic Field Vector (Tesla) for Maglev/Lorentz
        # Assuming env_data["magnetic_field"] is in microTesla (uT)
        b_scalar_uT = env_data.get("magnetic_field", 0.0)
        # Default direction: North (Y-axis) -> [0, B, 0]
        # In future, this could rotate based on latitude.
        env_data["magnetic_field_vec_T"] = [0.0, b_scalar_uT * 1e-6, 0.0]
            
        return env_data

    def _determine_regime(self, intent: str, env_type: str) -> str:
        """Classify vehicle into AERIAL, GROUND, MARINE, or SPACE."""
        
        # Explicit overrides (Priority: SPACE > MARINE > AERIAL > GROUND)
        if re.search(r'\b(space|orbit|satellite|rocket|probe|station|lander)\b', intent):
            return "SPACE"
        if re.search(r'\b(marine|boat|ship|sub|submarine|naval|underwater|sea|diver)\b', intent):
            return "MARINE"
        if re.search(r'\b(aerial|drone|fly|plane|jet|rotor|vtol|copter|glider|balloon|airship)\b', intent):
            return "AERIAL"
        if re.search(r'\b(ground|rover|car|bike|crawler|tank|walker|truck|bot)\b', intent):
            return "GROUND"
            
        # Infer from Environment Type if vague
        if env_type in ["AERO", "HURRICANE", "VOLCANO", "VENUS", "TITAN"]:
            # Default for thick atmos bodies is often aerial if not specified, 
            # but usually rovers are safer. defaulting to aerial for Titan/Venus if ambiguous due to interest
            # Actually, standard fallback is usually Ground for surfaces.
            if "drone" in intent or "fly" in intent: return "AERIAL"
            return "GROUND" # Safer default
            
        if env_type in ["GROUND", "INDUSTRIAL", "MOON", "MARS", "EUROPA", "STATIC"]:
            return "GROUND"
            
        if env_type in ["NAVAL", "UNDERSEA"]:
            return "MARINE"
            
        if "ORBIT" in env_type or env_type in ["ASTEROID", "SPACE"]:
            return "SPACE"
            
        return "GROUND" # Ultimate fallback

    def _determine_location(self, intent_lower: str) -> dict:
        """Determine physical location constants using regex."""
        
        # --- Solar System Bodies ---
        if re.search(r'\b(moon|lunar)\b', intent_lower):
            return self._moon_environment()
        if re.search(r'\b(mars|martian)\b', intent_lower):
            return self._mars_environment()
        if re.search(r'\b(venus|venusian)\b', intent_lower):
            return self._venus_environment()
        if re.search(r'\b(titan)\b', intent_lower):
            return self._titan_environment()
        if re.search(r'\b(europa)\b', intent_lower):
            return self._europa_environment()
        if re.search(r'\b(jupiter|jovian)\b', intent_lower):
            return self._jupiter_orbit_environment()
        if re.search(r'\b(saturn)\b', intent_lower):
            return self._saturn_orbit_environment()
        if re.search(r'\b(asteroid|comet|meteor)\b', intent_lower):
            return self._asteroid_environment()
            
        # --- Extreme Earth Environments ---
        if re.search(r'\b(volcano|lava|magma)\b', intent_lower):
            return self._volcano_environment()
        if re.search(r'\b(hurricane|storm|typhoon|tornado)\b', intent_lower):
            return self._hurricane_environment()
        if re.search(r'\b(undersea|underwater|depths?|ocean|trench)\b', intent_lower):
            return self._undersea_environment()
            
        # --- Standard Earth ---
        if re.search(r'\b(space|zero-g|orbit|vacuum)\b', intent_lower):
            return self._asteroid_environment() # Generic Space
        if re.search(r'\b(aero|air|sky|flight|cloud)\b', intent_lower):
            return self._aero_environment()
        if re.search(r'\b(naval|water|lake|river|sea)\b', intent_lower):
            return self._naval_environment()
        if re.search(r'\b(bio|medical|blood|vein)\b', intent_lower):
            return self._bio_environment()
        if re.search(r'\b(factory|industrial|warehouse|indoor)\b', intent_lower):
            return self._industrial_environment()
            
        # Default
        return self._ground_environment()

    # --- Environment Definitions ---

    def _moon_environment(self) -> dict:
        return {
            "type": "MOON",
            "gravity": 1.62, "fluid_density": 0.0, "pressure": 0.0, "temperature": -23.0,
            "viscosity": 0.0, "magnetic_field": 0.0, "solar_flux": 1361.0,
            "description": "Lunar surface. Vacuum, dust hazards, high solar flux (day)."
        }
    
    def _mars_environment(self) -> dict:
        return {
            "type": "MARS",
            "gravity": 3.71, "fluid_density": 0.020, "pressure": 600.0, "temperature": -63.0,
            "viscosity": 1.4e-5, "magnetic_field": 0.0, "solar_flux": 589.0, # ~43% of Earth
            "description": "Martian surface. Thin CO2 atmosphere, dust storms, cold."
        }
        
    def _venus_environment(self) -> dict:
        return {
            "type": "VENUS",
            "gravity": 8.87, "fluid_density": 65.0, "pressure": 9200000.0, "temperature": 462.0,
            "viscosity": 3.0e-5, "magnetic_field": 0.0, "solar_flux": 2600.0, # High albedo affects surface
            "description": "Venusian surface. Crushing pressure, lead-melting heat, acidic."
        }
        
    def _titan_environment(self) -> dict:
        return {
            "type": "TITAN",
            "gravity": 1.35, "fluid_density": 5.3, "pressure": 146700.0, "temperature": -179.0,
            "viscosity": 6.0e-6, "magnetic_field": 0.0, "solar_flux": 15.0, # ~1% Earth
            "description": "Titan surface. Thick nitrogen/methane atmosphere, low gravity, hydrocarbon lakes."
        }

    def _europa_environment(self) -> dict:
        return {
            "type": "EUROPA",
            "gravity": 1.31, "fluid_density": 0.0, "pressure": 0.0, "temperature": -160.0,
            "viscosity": 0.0, "magnetic_field": 400.0, # Induced field
            "solar_flux": 50.0,
            "description": "Europa surface. Ice shell, high radiation environment from Jupiter."
        }
        
    def _jupiter_orbit_environment(self) -> dict:
        return {
            "type": "JUPITER_ORBIT",
            "gravity": 24.79, "fluid_density": 0.0, "pressure": 0.0, "temperature": -145.0,
            "viscosity": 0.0, "magnetic_field": 420000.0, # Huge field (4.2 Gauss)
            "solar_flux": 50.0,
            "description": "Jupiter Orbit. Intense radiation belts, powerful magnetic field."
        }
        
    def _saturn_orbit_environment(self) -> dict:
        return {
            "type": "SATURN_ORBIT",
            "gravity": 10.44, "fluid_density": 0.0, "pressure": 0.0, "temperature": -178.0,
            "viscosity": 0.0, "magnetic_field": 20000.0,
            "solar_flux": 15.0,
            "description": "Saturn Orbit. Ring system hazards, complex gravity/moon interactions."
        }
    
    def _asteroid_environment(self) -> dict:
        return {
            "type": "ASTEROID",
            "gravity": 0.0, "fluid_density": 0.0, "pressure": 0.0, "temperature": -270.0,
            "viscosity": 0.0, "magnetic_field": 0.0, "solar_flux": 1361.0,
            "description": "Deep space/Asteroid microgravity. Vacuum, debris hazards."
        }
    
    def _volcano_environment(self) -> dict:
        return {
            "type": "VOLCANO",
            "gravity": 9.81, "fluid_density": 0.6, "pressure": 101325.0, "temperature": 800.0,
            "viscosity": 4.0e-5, "magnetic_field": 50.0, "solar_flux": 1000.0,
            "description": "Extreme heat environment. Ash particulates, corrosive gases."
        }
    
    def _hurricane_environment(self) -> dict:
        return {
            "type": "HURRICANE",
            "gravity": 9.81, "fluid_density": 1.225, "pressure": 95000.0, "temperature": 27.0,
            "wind_speed": 70.0, "viscosity": 1.8e-5, "magnetic_field": 50.0, "solar_flux": 200.0, # Cloudy
            "description": "Extreme wind, rain, turbulence."
        }
    
    def _undersea_environment(self) -> dict:
        return {
            "type": "UNDERSEA",
            "gravity": 9.81, "fluid_density": 1025.0, "pressure": 10000000.0, "temperature": 4.0,
            "viscosity": 1.0e-3, "magnetic_field": 50.0, "solar_flux": 0.0,
            "description": "High pressure deep ocean. No light, high salinity."
        }
    
    def _aero_environment(self) -> dict:
        return {
            "type": "AERO",
            "gravity": 9.81, "fluid_density": 1.225, "pressure": 101325.0, "temperature": 15.0,
            "viscosity": 1.8e-5, "magnetic_field": 50.0, "solar_flux": 1000.0,
            "description": "Standard Earth Atmosphere (ISA)."
        }
    
    def _naval_environment(self) -> dict:
        return {
            "type": "NAVAL",
            "gravity": 9.81, "fluid_density": 1025.0, "pressure": 101325.0, "temperature": 15.0,
            "viscosity": 1.0e-3, "magnetic_field": 50.0, "solar_flux": 1000.0,
            "description": "Surface water operations. Wave action, salt corrosion."
        }
    
    def _ground_environment(self) -> dict:
        return {
            "type": "GROUND",
            "gravity": 9.81, "fluid_density": 1.225, "pressure": 101325.0, "temperature": 20.0,
            "viscosity": 1.8e-5, "magnetic_field": 50.0, "solar_flux": 1000.0,
            "description": "Standard Earth Ground."
        }

    def _industrial_environment(self) -> dict:
        return {
            "type": "INDUSTRIAL",
            "gravity": 9.81, "fluid_density": 1.225, "pressure": 101325.0, "temperature": 22.0,
            "viscosity": 1.8e-5, "magnetic_field": 50.0, "solar_flux": 0.0, # Artificial light
            "description": "Controlled indoor industrial."
        }

    def _bio_environment(self) -> dict:
        return {
            "type": "BIO",
            "gravity": 9.81, "fluid_density": 1050.0, "pressure": 101325.0, "temperature": 37.0,
            "viscosity": 3.0e-3, "magnetic_field": 0.0, "solar_flux": 0.0,
            "description": "Internal biological. Non-newtonian fluids likely."
        }

    # --- Phase 20: Swarm Environment Logic ---
    
    def init_swarm_resources(self, width: float = 100.0, height: float = 100.0, count: int = 10) -> List[Dict[str, Any]]:
        """
        Generate random resource piles for Swarm Simulation.
        """
        import random
        piles = []
        for _ in range(count):
            piles.append({
                "id": f"res_{random.randint(1000,9999)}",
                "x": random.uniform(-width/2, width/2),
                "y": random.uniform(-height/2, height/2),
                "type": random.choice(["ORE", "ENERGY"]),
                "amount": random.uniform(100.0, 1000.0),
                "radius": 5.0
            })
        return piles

    def update_pheromones(self, pheromone_grid: Dict[str, float], decay_rate: float = 0.95) -> Dict[str, float]:
        """
        Apply decay to pheromone grid (Entropy).
        """
        new_grid = {}
        for key, value in pheromone_grid.items():
            new_val = value * decay_rate
            if new_val > 0.01: # Cull weak signals
                new_grid[key] = new_val
        return new_grid

    def consume_resource(self, piles: List[Dict], agent_pos: List[float], amount: float) -> float:
        """
        Attempt to harvest resources near agent. returns amount harvested.
        """
        harvested = 0.0
        import math
        
        ax, ay = agent_pos[0], agent_pos[1]
        
        for pile in piles:
            dx = pile["x"] - ax
            dy = pile["y"] - ay
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < (pile["radius"] + 1.0): # Within range
                available = pile["amount"]
                take = min(available, amount)
                pile["amount"] -= take
                harvested += take
                if harvested >= amount:
                    break
                    
                if harvested >= amount:
                    break
                    
        return harvested

    def evaluate_terrain_sdf(self, position: List[float], terrain_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query exact terrain distance using VMK.
        Positive = In Air, Negative = Underground/In Obstacle.
        """
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
        except ImportError:
            return {"error": "VMK not available"}

        # Optimization: Caching kernel? For MVP, we recreate.
        stock_dims = terrain_map.get("dims", [1000, 1000, 1000])
        kernel = SymbolicMachiningKernel(stock_dims=stock_dims)
        
        registered = set()
        for op in terrain_map.get("obstacles", []):
            tid = op.get("tool_id", "terrain_feature")
            if tid not in registered:
                # Default "Hill / Rock" radius
                kernel.register_tool(ToolProfile(id=tid, radius=op.get("radius", 10.0), type="BALL"))
                registered.add(tid)
            kernel.execute_gcode(VMKInstruction(**op))
            
        p = np.array(position)
        sdf = kernel.get_sdf(p)
        
        # Interpret SDF:
        # Our VMK is "Stock - Cuts".
        # If Terrain is "Cuts into ground" (Caves/Canyons)?
        # If Terrain is "Additions to ground"? VMK is subtractive.
        # To model Hills:
        # We start with huge Stock (Ground).
        # We cut away "Air" to leave Hills? That's complex sculpting.
        # Alternative: We treat "Stock" as Air, and "Cuts" as Obstacles (Collision volumes)?
        # PhysicsAgent collision check treated Stock as Box (Solid) and Cuts as Tunnels (Air).
        # If we want Hills involved:
        # Modeling positive features in subtractive kernel:
        # Start with Block. Cut everything EXCEPT the hill.
        # Or, just use SDF primitives directly if we expand Kernel to support Union.
        # For now, implemented as Subtractive: "Terrain" = Solid Block with Tunnels/Caves.
        # So SDF < 0 is "Inside Rock". SDF > 0 is "In Air".
        
        # Return distance to surface
        return {
            "sdf": sdf,
            "is_underground": sdf < 0,
            "distance_to_surface": abs(sdf),
            "gradient": [0,0,1] # Placeholder for future normal vector calc
        }

    def _initialize_oracles(self):
        """Initialize Oracles for environmental analysis"""
        try:
            from agents.chemistry_oracle.chemistry_oracle import ChemistryOracle
            from agents.physics_oracle.physics_oracle import PhysicsOracle
            self.chemistry_oracle = ChemistryOracle()
            self.physics_oracle = PhysicsOracle()
            self.has_oracles = True
        except ImportError:
            self.chemistry_oracle = None
            self.physics_oracle = None
            self.has_oracles = False

    def analyze_emissions_oracle(self, params: dict) -> dict:
        """Analyze chemical emissions using Chemistry Oracle"""
        if not hasattr(self, 'has_oracles'):
            self._initialize_oracles()
        
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.chemistry_oracle.solve(
            query="Emissions analysis",
            domain="KINETICS",
            params=params
        )
    
    def analyze_atmospheric_dispersion_oracle(self, params: dict) -> dict:
        """Analyze atmospheric dispersion using Physics Oracle (FLUID)"""
        if not hasattr(self, 'has_oracles'):
            self._initialize_oracles()
        
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.physics_oracle.solve(
            query="Atmospheric dispersion",
            domain="FLUID",
            params=params
        )
