from enum import Enum

class CompileMode(Enum):
    """Compilation quality levels"""
    PREVIEW = "preview"      # Fastest, lowest quality (Manifold only)
    STANDARD = "standard"    # Balanced (Hybrid)
    EXPORT = "export"        # Highest quality (OCCT)


class GeometryFormat(Enum):
    """Export formats"""
    GLTF = "gltf"
    STL = "stl"
    STEP = "step"
    IGES = "iges"
