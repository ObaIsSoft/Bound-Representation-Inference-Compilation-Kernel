import requests
import os

url = "https://raw.githubusercontent.com/nasa/NASA-3D-Resources/master/3D%20Models/NASA%203D%20Models/Space%20Shuttle/Space%20Shuttle.stl"

try:
    print(f"Attempting download from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    print("Download connection established.")
    
    with open("test_shuttle.stl", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print(f"Download complete. Size: {os.path.getsize('test_shuttle.stl')} bytes")
    os.remove("test_shuttle.stl")
    print("Cleanup complete. Success.")
except Exception as e:
    print(f"FAILED: {e}")
