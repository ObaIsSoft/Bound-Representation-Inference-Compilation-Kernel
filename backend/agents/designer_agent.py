from typing import Dict, Any, List
import logging
import colorsys
import random

logger = logging.getLogger(__name__)

class DesignerAgent:
    """
    Designer Agent - Procedural Aesthetics.
    
    Determines aesthetics, colorways, and materials using procedural generation logic.
    Supports color theory harmonization (Complementary, Split, Triadic).
    """
    
    def __init__(self):
        self.name = "DesignerAgent"
        self.db_path = "data/materials.db"

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate design language.
        
        Args:
            params: {
                "style": str (optional),
                "base_color": str (hex, optional),
                "harmony": str (complementary/split/analogous/industrial)
            }
        """
        logger.info(f"{self.name} applying design language...")
        
        style = params.get("style", "generative").lower()
        base_color_hex = params.get("base_color", None) # e.g. "#FF5733"
        harmony = params.get("harmony", "complementary")
        
        logs = [f"[DESIGNER] Generating palette for style: {style}"]
        
        if style == "industrial" and not base_color_hex:
             return self._run_classic_mode(params)
        
        # 1. Procedural Color Generation (Learned Bias)
        if not base_color_hex:
            # Load preferences
            prefs = self._load_preferences()
            preferred_hues = prefs.get("hues", []) # List of (hue, weight)
            
            # Epsilon-Greedy: 20% random exploration, 80% exploitation of top hues
            if random.random() < 0.2 or not preferred_hues:
                h = random.random()
                logs.append("[DESIGNER] Exploration Mode (Random Hue)")
            else:
                # Weighted choice
                hues, weights = zip(*preferred_hues)
                h = random.choices(hues, weights=weights, k=1)[0]
                # Add small jitter
                h = (h + random.uniform(-0.05, 0.05)) % 1.0
                logs.append(f"[DESIGNER] Exploitation Mode (Learned Hue {h:.2f})")

            s = random.uniform(0.5, 0.9)
            v = random.uniform(0.6, 0.95)
            base_color_hex = self._hsv_to_hex(h, s, v)
        else:
            h, s, v = self._hex_to_hsv(base_color_hex)

        # 2. Generate Harmony
        palette = self._generate_harmony(h, s, v, harmony)
        logs.append(f"[DESIGNER] Applied harmony: {harmony}")
        
        # 3. Generate Material Properties (Procedural)
        # Infer material vibe from style
        material_props = self._generate_material_props(style)
        
        selected = {
            "primary": palette["primary"],
            "accent": palette["accent"],
            "secondary": palette["secondary"],
            "finish": material_props["finish"],
            "roughness": material_props["roughness"],
            "metalness": material_props["metalness"],
            "description": f"Generated {harmony} palette with {material_props['finish']} finish."
        }
        
        return {
            "status": "success",
            "aesthetics": selected,
            "primitives": self._generate_primitives(style, harmony), # New Capability
            "logs": logs
        }
        
    def _generate_primitives(self, style: str, harmony: str) -> List[Dict]:
        """
        Generates optional semantic sketch primitives based on style.
        Allows the agent to 'draw' initial concepts.
        """
        # Load style config if needed (or rely on what was loaded in _generate_material_props)
        # To be efficient, we should load it once or cache it.
        # But _generate_material_props loads it fresh every time.
        
        # Let's reuse the loading logic or just call it to get the map.
        # Ideally, we should refactor to load in __init__, but sticking to the current pattern:
        
        primitives = []
        
        # We need the full map, not just the props for one style.
        # So we repeat the loading logic briefly here or abstract it.
        import json
        import os
        
        config_path = os.path.join(os.path.dirname(__file__), "../data/standards_config.json")
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                styles_map = data.get("designer_styles", {})
                
            # Iterate through all styles in the map
            # If the current 'style' param matches a key in the map (partial match),
            # add its primitives if they exist.
            
            for key, props in styles_map.items():
                if key in style:
                    if "primitives" in props:
                        primitives.extend(props["primitives"])
                        
        except Exception as e:
            logger.error(f"Failed to load primitives: {e}")
            
        return primitives
        
    def _generate_harmony(self, h, s, v, type="complementary") -> Dict:
        """Generate color harmony."""
        primary = self._hsv_to_hex(h, s, v)
        
        if type == "complementary":
            h2 = (h + 0.5) % 1.0
            accent = self._hsv_to_hex(h2, s, v)
            secondary = self._hsv_to_hex(h, max(0, s-0.3), min(1, v+0.2)) # lighter primary
            
        elif type == "split":
            h2 = (h + 0.4) % 1.0
            h3 = (h + 0.6) % 1.0
            accent = self._hsv_to_hex(h2, s, v)
            secondary = self._hsv_to_hex(h3, s, v)
            
        elif type == "analogous":
            h2 = (h + 0.08) % 1.0
            accent = self._hsv_to_hex(h2, s, v)
            secondary = self._hsv_to_hex((h - 0.08)%1.0, s, v)
            
        else: # Monochromatic default
            accent = self._hsv_to_hex(h, s, max(0, v-0.4))
            secondary = self._hsv_to_hex(h, max(0, s-0.5), v)
            
        return {"primary": primary, "accent": accent, "secondary": secondary}

    def _generate_material_props(self, style: str) -> Dict:
        """Infer material properties from style keywords using external config."""
        import json
        import os
        
        config_path = os.path.join(os.path.dirname(__file__), "../data/standards_config.json")
        styles_map = {}
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                styles_map = data.get("designer_styles", {})
        except Exception as e:
            logger.error(f"Failed to load designer styles: {e}")
            return {"finish": "standard", "roughness": 0.5, "metalness": 0.5}

        # Check for keywords in style string
        for key, props in styles_map.items():
            if key in style:
                return props
                
        return styles_map.get("default", {"finish": "standard", "roughness": 0.5, "metalness": 0.5})

    def _hex_to_hsv(self, hex_code):
        hex_code = hex_code.lstrip('#')
        r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        return colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)

    def _hsv_to_hex(self, h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

    def _run_classic_mode(self, params):
        # Fallback to older SQLite logic (abbreviated)
        return {
            "status": "success", 
            "aesthetics": {"primary": "#333", "accent": "#f00", "finish": "classic"},
            "logs": ["[DESIGNER] Used fallback industrial mode"]
        }

    def _load_preferences(self) -> Dict:
        """Load learned design preferences."""
        import json
        import os
        path = "data/designer_agent_weights.json"
        if not os.path.exists(path): return {}
        try:
            with open(path, 'r') as f: return json.load(f)
        except: return {}

    def update_preferences(self, successful_hue: float):
        """Reinforce a successful design choice."""
        import json
        import os
        path = "data/designer_agent_weights.json"
        
        prefs = {}
        if os.path.exists(path):
            try: 
                with open(path, 'r') as f: 
                    prefs = json.load(f)
            except: 
                pass
            
        current_hues = prefs.get("hues", [])
        
        # Crude "clustering" update: If close to existing hue, boost weight. Else add new.
        found = False
        new_hues = []
        for h, w in current_hues:
            if abs(h - successful_hue) < 0.1: # Similarity threshold
                new_hues.append([h, w + 1.0]) # Boost weight
                found = True
            else:
                new_hues.append([h, w])
        
        if not found:
            new_hues.append([successful_hue, 1.0])
            
        # Limit memory size
        if len(new_hues) > 20:
             # Keep top 20 by weight
             new_hues.sort(key=lambda x: x[1], reverse=True)
             new_hues = new_hues[:20]
             
        prefs["hues"] = new_hues
        
        with open(path, 'w') as f: json.dump(prefs, f, indent=2)
