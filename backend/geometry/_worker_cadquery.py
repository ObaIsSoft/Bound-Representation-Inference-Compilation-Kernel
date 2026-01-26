import sys
import json
import cadquery as cq
import logging

# Configure logging to stderr to avoid polluting stdout (used for result IPC)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("CQ_Worker")

def build_model(tree):
    """
    Reconstructs the CadQuery model from the recipe tree.
    """
    # Start with empty Workplane? 
    # CQ usually works by accumulation or specific boolean ops.
    # We'll use a functional approach: Create shapes -> Boolean them.
    
    combined = None
    
    for node in tree:
        shape = create_primitive(node)
        
        # Transform
        node_transform = node.get("transform", {})
        if node_transform:
             # Translate
             trans = node_transform.get("translate", [0,0,0])
             shape = shape.translate((trans[0], trans[1], trans[2]))
             # Rotate?
             
        if combined is None:
            combined = shape
        else:
            op = node.get("operation", "UNION").upper()
            if op == "UNION":
                combined = combined.union(shape)
            elif op == "SUBTRACT" or op == "DIFFERENCE":
                combined = combined.cut(shape)
            elif op == "INTERSECT":
                combined = combined.intersect(shape)
                
    return combined

def create_primitive(node):
    ptype = node.get("type", "box")
    params = node.get("params", {})
    
    if ptype == "box":
        l = params.get("length", 1.0)
        w = params.get("width", 1.0)
        h = params.get("height", 1.0)
        return cq.Workplane("XY").box(l, w, h)
        
    elif ptype == "cylinder":
        h = params.get("height", 1.0)
        r = params.get("radius", 1.0)
        return cq.Workplane("XY").cylinder(h, r)
        
    elif ptype == "sphere":
        r = params.get("radius", 1.0)
        return cq.Workplane("XY").sphere(r)
        
    # Fallback
    return cq.Workplane("XY").box(0.1, 0.1, 0.1)

def main():
    try:
        # Read JSON recipe from stdin
        input_data = sys.stdin.read()
        if not input_data:
            logger.error("No input data received")
            sys.exit(1)
            
        request = json.loads(input_data)
        tree = request.get("tree", [])
        output_format = request.get("output_format", "step")
        req_id = request.get("request_id", "unknown")
        
        logger.info(f"Building Request {req_id} with {len(tree)} nodes...")
        
        result_obj = build_model(tree)
        
        # Export
        import os
        filename = f"data/exports/{req_id}.{output_format}"
        os.makedirs("data/exports", exist_ok=True)
        
        if output_format.lower() == "step":
            cq.exporters.export(result_obj, filename, "STEP")
        elif output_format.lower() == "stl":
            cq.exporters.export(result_obj, filename, "STL")
        else:
             logger.error(f"Unknown format: {output_format}")
             sys.exit(1)
             
        # Return result to parent process: JSON on stdout
        response = {
            "success": True,
            "file_path": os.path.abspath(filename)
        }
        print(json.dumps(response))
        
    except Exception as e:
        logger.error(f"Worker Failed: {e}")
        # Return error JSON
        err_response = {"success": False, "error": str(e)}
        print(json.dumps(err_response))
        sys.exit(1)

if __name__ == "__main__":
    main()
