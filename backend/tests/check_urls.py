import requests

candidates = [
    "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/cube.stl",
    "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj",
    "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/bunny.obj",
    "https://raw.githubusercontent.com/nasa/NASA-3D-Resources/master/3D%20Models/NASA%203D%20Models/Space%20Shuttle/Space%20Shuttle.stl"
]

for url in candidates:
    try:
        r = requests.head(url)
        print(f"{r.status_code} : {url}")
    except:
        print(f"ERR : {url}")
