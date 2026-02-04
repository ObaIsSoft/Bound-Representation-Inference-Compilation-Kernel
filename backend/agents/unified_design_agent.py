import random
import hashlib
import numpy as np
import copy
import logging
import colorsys
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# --- GENOMIC FOUNDATION ---

class StyleDimension(Enum):
    """The wide spectrum of aesthetic design"""
    EPOCH = "epoch"
    REGION = "region"
    GENRE = "genre"
    SILHOUETTE = "silhouette"
    MASSING = "massing"
    TOPOLOGY = "topology"
    SUBSTANCE = "substance"
    SURFACE_AGE = "surface_age"
    CRAFT_SIGNAL = "craft_signal"
    MOOD = "mood"
    ENERGY = "energy"
    SCALE_PERCEPTION = "scale"
    READABILITY = "readability"
    DETAIL_DENSITY = "detail_density"
    COLOR_STRATEGY = "color_strategy"

@dataclass 
class StyleGene:
    dimension: StyleDimension
    value: float
    confidence: float = 1.0
    provenance: str = "inferred"

@dataclass
class ColorGene:
    h: float = 0.0
    s: float = 0.0
    v: float = 1.0
    role: str = "primary"
    
    def to_hex(self) -> str:
        rgb = colorsys.hsv_to_rgb(self.h, self.s, self.v)
        return '#%02x%02x%02x' % tuple(int(x*255) for x in rgb)

@dataclass
class MaterialGene:
    roughness: float = 0.5
    metalness: float = 0.0
    type: str = "standard"
    name: Optional[str] = None
    physics_props: Dict[str, Any] = field(default_factory=dict)
    
    def ground_physics(self):
        """Bridge style to physical reality using MaterialAgent"""
        if not self.name: return
        try:
            from agents.material_agent import MaterialAgent
            agent = MaterialAgent()
            result = agent.run(self.name)
            if "properties" in result:
                self.physics_props = result["properties"]
        except Exception as e:
            logger.error(f"Material grounding failed: {e}")

    def to_pbr(self) -> Dict[str, Any]:
        return {
            "roughness": self.roughness,
            "metalness": self.metalness,
            "base_type": self.type,
            "material_name": self.name,
            "physics": self.physics_props
        }

@dataclass
class DesignGenome3D:
    id: str = field(default_factory=lambda: f"3d_{random.randint(10000,99999)}")
    style_genes: Dict[StyleDimension, StyleGene] = field(default_factory=dict)
    color_genes: List[ColorGene] = field(default_factory=list)
    material_genes: List[MaterialGene] = field(default_factory=list)
    geometry_params: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    
    def get_stable_seed(self) -> int:
        h = hashlib.sha256(self.id.encode())
        return int(h.hexdigest(), 16) % (2**32)

    def to_3d_generation_payload(self) -> Dict:
        return {
            "id": self.id,
            "prompt": self.reconstruct_prompt(),
            "style_vector": [self.style_genes.get(d, StyleGene(d, 0.5)).value for d in StyleDimension],
            "geometry_params": self.geometry_params,
            "colors": [c.to_hex() for c in self.color_genes],
            "material_overrides": [m.to_pbr() for m in self.material_genes],
            "seed": self.get_stable_seed()
        }

    def reconstruct_prompt(self) -> str:
        parts = []
        if StyleDimension.EPOCH in self.style_genes:
            parts.append(self._quantize(self.style_genes[StyleDimension.EPOCH].value, ["Ancient", "Industrial", "Modernist", "Futuristic", "Cyberpunk"]))
        if StyleDimension.MOOD in self.style_genes:
            parts.append(self._quantize(self.style_genes[StyleDimension.MOOD].value, ["Aggressive", "Clean", "Ethereal"]))
        return " ".join(parts) + " 3D asset"

    def _quantize(self, val: float, labels: List[str]) -> str:
        idx = min(int(val * len(labels)), len(labels) - 1)
        return labels[idx]

# --- INTELLIGENCE LAYER ---

class PromptInterpreter:
    EPOCH_ANCHORS = {"cyberpunk": 0.85, "neon": 0.9, "retro": 0.3, "futuristic": 0.9, "ancient": 0.05, "brutalist": 0.35, "minimalist": 0.5}
    MOOD_ANCHORS = {"ominous": 0.1, "peaceful": 0.9, "aggressive": 0.05, "elegant": 0.75}
    
    # Sense Layer: Direct Overrides
    COLOR_MAP = {
        "red": (0.0, 0.9, 0.8), "blue": (0.6, 0.8, 0.9), "green": (0.3, 0.7, 0.7),
        "yellow": (0.15, 0.8, 0.9), "purple": (0.8, 0.7, 0.8), "orange": (0.08, 0.8, 0.9),
        "white": (0.0, 0.0, 1.0), "black": (0.0, 0.0, 0.1), "gray": (0.0, 0.0, 0.5)
    }
    MAT_MAP = {
        "aluminum": "Aluminum 6061", "steel": "Steel 4130", "plastic": "ABS Plastic", 
        "wood": "Oak Wood", "titanium": "Titanium Alloy", "carbon": "Carbon Fiber"
    }
    FINISH_MAP = {
        "rough": {"roughness": 0.9, "metalness": 0.0},
        "smooth": {"roughness": 0.2, "metalness": 0.0},
        "shiny": {"roughness": 0.1, "metalness": 0.8},
        "metallic": {"roughness": 0.3, "metalness": 1.0},
        "matte": {"roughness": 0.95, "metalness": 0.0}
    }

    _mat_agent = None

    @property
    def mat_agent(self):
        if PromptInterpreter._mat_agent is None:
            from agents.material_agent import MaterialAgent
            PromptInterpreter._mat_agent = MaterialAgent()
        return PromptInterpreter._mat_agent

    def parse(self, prompt: str) -> DesignGenome3D:
        p = prompt.lower()
        genome = DesignGenome3D()
        
        # 1. Stylistic Inference (Genes)
        for dim, anchors in [(StyleDimension.EPOCH, self.EPOCH_ANCHORS), (StyleDimension.MOOD, self.MOOD_ANCHORS)]:
            matches = [v for k, v in anchors.items() if k in p]
            val = sum(matches)/len(matches) if matches else 0.5
            genome.style_genes[dim] = StyleGene(dim, val, confidence=0.8 if matches else 0.3)

        # 2. Sense Layer: Color Overrides
        for c, hsv in self.COLOR_MAP.items():
            if c in p:
                genome.color_genes = [ColorGene(*hsv)]
                break
        # 3. Sense Layer: Material & Finish Overrides
        mat_gene = MaterialGene()
        
        # 3.1 Hardcoded Keywords
        found_mat = False
        for key, full_name in self.MAT_MAP.items():
            if key in p:
                mat_gene.name = full_name
                mat_gene.type = key
                try:
                    mat_res = self.mat_agent.run(full_name)
                    mat_gene.physics_props = mat_res.get("properties", {})
                except Exception as e:
                    logger.error(f"Failed to ground hardcoded material {full_name}: {e}")
                found_mat = True
                break
                
        # 3.2 Dynamic Pattern Detection (Formulas & Specific Grades)
        if not found_mat:
            import re
            patterns = [
                r"\b([A-Z][a-z]?\d+([A-Z][a-z]?\d*)*)\b", # Formulas
                r"\b([A-Z]{1,2})\b", # Elemental symbols
                r"\b(grade \d+|[0-9]{4}-[A-Z0-9]+|[0-9]{3}[L]?)\b", # Alloy grades
                r"\b(ti-6al-4v|inconel \d+)\b" # Explicit alloys
            ]
            for pattern in patterns:
                matches = re.finditer(pattern, prompt)
                best_match = None
                for match in matches:
                    text = match.group(0)
                    if text.upper() == "3D": continue
                    if not best_match or len(text) > len(best_match):
                        best_match = text
                
                if best_match:
                    mat_gene.name = best_match
                    mat_gene.type = "standard"
                    try:
                        mat_res = self.mat_agent.run(best_match)
                        if mat_res and "error" not in mat_res:
                            mat_gene.name = mat_res.get("name", best_match)
                            mat_gene.physics_props = mat_res.get("properties", {})
                            break
                    except Exception as e:
                        logger.error(f"Failed to ground dynamic material {best_match}: {e}")
                        continue

        # 3.3 Finish Overrides
        for f, props in self.FINISH_MAP.items():
            if f in p:
                mat_gene.roughness = props.get("roughness", mat_gene.roughness)
                mat_gene.metalness = props.get("metalness", mat_gene.metalness)
                if f == "metallic" and not mat_gene.type:
                    mat_gene.type = "metal"
        
        # Trigger grounding in a safer, lazy way if we have a name
        # We will do this via the mat_gene's own lazy property if implemented,
        # but for now we just return the detected name.
        genome.material_genes = [mat_gene]
        return genome

class DesignExplorer3D:
    def __init__(self, base: DesignGenome3D):
        self.base = base
        self.population = []

    def explore(self, count: int = 16) -> List[DesignGenome3D]:
        self.population = []
        for i in range(count):
            g = copy.deepcopy(self.base)
            g.id = f"{self.base.id}_var_{i}"
            for dim in StyleDimension:
                # LHS-ish shuffle
                val = (self.base.style_genes.get(dim, StyleGene(dim, 0.5)).value * 0.4) + (random.random() * 0.6)
                g.style_genes[dim] = StyleGene(dim, np.clip(val, 0, 1))
            self.population.append(g)
        return self.population

    def user_select(self, genome_id: str) -> Optional[DesignGenome3D]:
        """Find a variant by ID in the current population"""
        for g in self.population:
            if g.id == genome_id:
                return g
        return None

    def evolve(self, parent_ids: List[str], count: int = 16) -> List[DesignGenome3D]:
        """Breeding: Create new variants based on selected parents (GA style)"""
        # Find the parent genomes
        parents = [self.user_select(pid) for pid in parent_ids]
        parents = [p for p in parents if p]
        
        # If no parents found, fallback to fresh explore (reshuffle)
        if not parents:
            return self.explore(count)
            
        self.population = []
        for i in range(count):
            # 1. Crossover: Pick two parents
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            
            child = copy.deepcopy(p1)
            child.id = f"{self.base.id}_evol_{i}"
            child.generation = max(p1.generation, p2.generation) + 1
            
            # 2. Gene Mixing (Style genes)
            for dim in StyleDimension:
                # Randomly inherit from p1 or p2
                base_val = (p1.style_genes.get(dim).value + p2.style_genes.get(dim).value) / 2
                # Add mutation noise (10%)
                mutation = random.uniform(-0.1, 0.1)
                child.style_genes[dim] = StyleGene(dim, np.clip(base_val + mutation, 0, 1))
                
            self.population.append(child)
        return self.population

# --- QUALITY LAYER ---

class QualityRefiner:
    def refine(self, geometry_tree: List[Dict], target: str = "high") -> Dict[str, Any]:
        """Integrated quality enhancement logic"""
        levels = {"low": 1, "medium": 2, "high": 3, "ultra": 4}
        subdivisions = levels.get(target, 2)
        
        return {
            "refinement_level": subdivisions,
            "smoothing": True if subdivisions > 1 else False,
            "quality_score": subdivisions / 4.0
        }

# --- UNIFIED AGENT ---

class UnifiedDesignAgent:
    """The Single Point of Contact for Design in BRICK OS"""
    
    def __init__(self):
        self.interpreter = PromptInterpreter()
        self.refiner = QualityRefiner()
        self.active_explorer: Optional[DesignExplorer3D] = None

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for the agent.
        Mode: 'interpret', 'explore', or 'refine'
        """
        mode = params.get("mode", "interpret")
        prompt = params.get("prompt", "")
        
        if mode == "interpret":
            genome = self.interpreter.parse(prompt)
            return {
                "status": "success",
                "genome": genome.to_3d_generation_payload(),
                "thought": f"Created design genome for: {prompt}"
            }
            
        elif mode == "explore":
            genome = self.interpreter.parse(prompt)
            self.active_explorer = DesignExplorer3D(genome)
            variants = self.active_explorer.explore(count=params.get("count", 16))
            return {
                "status": "success",
                "variants": [v.to_3d_generation_payload() for v in variants]
            }

        elif mode == "evolve":
            if not self.active_explorer:
                return {"status": "error", "message": "No active exploration session to evolve from"}
            
            parent_ids = params.get("parent_ids", [])
            variants = self.active_explorer.evolve(parent_ids, count=params.get("count", 16))
            return {
                "status": "success",
                "variants": [v.to_3d_generation_payload() for v in variants],
                "thought": f"Bred {len(variants)} new variants from parents: {parent_ids}"
            }
            
        elif mode == "refine":
            geo = params.get("geometry_tree", [])
            quality = params.get("quality", "high")
            report = self.refiner.refine(geo, quality)
            return {
                "status": "success",
                "quality_report": report
            }
            
        return {"status": "error", "message": f"Unknown mode: {mode}"}
