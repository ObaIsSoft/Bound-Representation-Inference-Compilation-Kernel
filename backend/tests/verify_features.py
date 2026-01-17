import requests
import json

BASE_URL = "http://localhost:8000/api/components"

def test_inspect():
    print("--- Testing /inspect ---")
    url = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj"
    try:
        res = requests.post(f"{BASE_URL}/inspect", json={"url": url})
        print(f"Status: {res.status_code}")
        print(json.dumps(res.json(), indent=2))
    except Exception as e:
        print(f"Inspect Failed: {e}")

def test_smart_sourcing():
    print("\n--- Testing Smart Sourcing (Search: 'phone') ---")
    try:
        res = requests.get(f"{BASE_URL}/catalog?search=phone")
        data = res.json()
        catalog = data.get("catalog", [])
        print(f"Found {len(catalog)} items.")
        for item in catalog:
            print(f"- {item['name']} ({item['category']})")
            
        if len(catalog) >= 4 and "OLED" in catalog[0]['name']:
            print("SUCCESS: Phone kit returned.")
        else:
            print("FAIL: Did not get expected phone kit.")
            
    except Exception as e:
         print(f"Search Failed: {e}")

if __name__ == "__main__":
    test_inspect()
    test_smart_sourcing()
