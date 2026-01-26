import sys
import os
import asyncio
import json

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.hybrid_engine import HybridGeometryEngine

async def run_tests():
    print("🚀 Verifying Hybrid Geometry Architecture...")
    
    engine = HybridGeometryEngine()
    
    # Define Test Tree
    test_tree = [
        {
            "id": "box1",
            "type": "box",
            "params": {"length": 10, "width": 10, "height": 10}
        },
        {
            "id": "cyl1",
            "type": "cylinder",
            "params": {"radius": 2, "height": 20},
            "operation": "SUBTRACT",
            "transform": {"translate": [0,0,0]} 
        }
    ]
    
    # Test 1: Hot Path (Manifold -> GLB)
    print("\n[Test 1] Testing Hot Path (Manifold)...")
    res_hot = await engine.compile(test_tree, format="glb", request_id="test_hot_1")
    if res_hot.success:
        print(f"✅ Hot Path Successful! Payload Size: {len(res_hot.payload)} bytes")
        # Save for inspection
        with open("test_hot.glb", "wb") as f:
            f.write(res_hot.payload)
    else:
        print(f"❌ Hot Path Failed: {res_hot.error}")
        
    # Test 2: Cold Path (CadQuery -> STEP)
    print("\n[Test 2] Testing Cold Path (CadQuery Worker)...")
    res_cold = await engine.compile(test_tree, format="step", request_id="test_cold_1")
    if res_cold.success:
        print(f"✅ Cold Path Successful! File ID: {res_cold.file_path}")
        if os.path.exists(res_cold.file_path):
             print(f"   -> File exists: {os.path.getsize(res_cold.file_path)} bytes")
    else:
        print(f"❌ Cold Path Failed: {res_cold.error}")
        
    # Test 3: Unsupported Format
    print("\n[Test 3] Testing Routing Logic (Invalid Format)...")
    res_bad = await engine.compile(test_tree, format="obj") # Not implemented yet
    if not res_bad.success:
        print(f"✅ Router correctly rejected 'obj': {res_bad.error}")
    else:
        print("❌ Router accepted invalid format!")

def main():
    asyncio.run(run_tests())

if __name__ == "__main__":
    main()
