
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.component_agent import ComponentAgent
from database.supabase_client import SupabaseClientWrapper

def verify():
    print("🔍 -- Verifying Universal Component Catalog --")
    
    # 1. Check DB Connection
    db = SupabaseClientWrapper()
    if not db.enabled:
        print("⚠️ Supabase not enabled. Skipping integration tests.")
        return

    # 2. Check Data Population
    try:
        res = db.client.table("components").select("count", count="exact").execute()
        count = res.count
        print(f"📊 Catalog Size: {count} items")
        
        if count == 0:
            print("⚠️ Catalog is empty. Please run 'python3 backend/scripts/populate_universal_catalog.py'")
    except Exception as e:
        print(f"❌ DB Access Error (Schema mismatch?): {e}")

    # 3. Test Component Agent Search
    agent = ComponentAgent()
    
    # Test 1: Broad Search (Motors)
    print("\n[Test 1] Searching for Motors > 300W...")
    reqs = {"min_max_power_w": 300, "category": "motor"} # Note: key matches spec 'max_power_w' with 'min_' prefix
    result = agent.run({"requirements": reqs, "limit": 3})
    
    if result["count"] > 0:
        print(f"✅ Found {result['count']} matches.")
        for item in result["selection"]:
            print(f"   - {item['name']} | Power: {item['specs'].get('max_power_w')}W | Mass: {item['mass_g']:.1f}g")
    else:
        print("🔸 No matches found (or DB empty).")

    # Test 2: Stochastic Instantiation
    print("\n[Test 2] Testing Stochastic Variance (Volatility 0.1)...")
    if result["count"] > 0:
        # Re-run selection with volatility
        res_v = agent.run({"requirements": reqs, "limit": 1, "volatility": 0.1})
        p1 = res_v["selection"][0]
        
        # Run again
        res_v2 = agent.run({"requirements": reqs, "limit": 1, "volatility": 0.1})
        p2 = res_v2["selection"][0]
        
        print(f"   Instance 1 Mass: {p1['mass_g']:.4f}")
        print(f"   Instance 2 Mass: {p2['mass_g']:.4f}")
        
        if p1['mass_g'] != p2['mass_g']:
            print("✅ Variance Observed (Mass differs between instances)")
        else:
            print("⚠️ No Variance (Check distribution params)")

    # Test 3: Complex Component Retrieval (ICE)
    print("\n[Test 3] Searching for Internal Combustion Engine...")
    reqs = {"category": "internal_combustion_engine"}
    res_ice = agent.run({"requirements": reqs, "limit": 1})
    
    if res_ice["count"] > 0:
        ice = res_ice["selection"][0]
        print(f"✅ Found ICE: {ice['name']}")
        
        # Check Behavior Model
        if "behavior_model" in ice and ice["behavior_model"]:
            bm = ice["behavior_model"]
            print(f"   - Behavior Type: {bm.get('type')}")
            if "reliability" in bm:
                print(f"   - Reliability Data Found: MTBF {bm['reliability'].get('mtbf_hours')} hours")
        else:
            print("⚠️ No Behavior Model found on ICE")
    else:
        print("🔸 No ICE found (Did you re-run population?)")

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    verify()
