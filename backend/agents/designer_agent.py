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
             # Fallback to classic DB/industrial logic if strictly requested
             return self._run_classic_mode(params)
        
        # 1. Procedural Color Generation
        if not base_color_hex:
            # Generate a random base color if none provided
            # Bias towards nice saturation/value (avoid mud)
            h = random.random()
            s = random.uniform(0.5, 0.9)
            v = random.uniform(0.6, 0.95)
            # base_color_rgb = colorsys.hsv_to_rgb(h, s, v)
            base_color_hex = self._hsv_to_hex(h, s, v)
            logs.append(f"[DESIGNER] Generated base color: {base_color_hex}")
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
            "logs": logs
        }
        
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
