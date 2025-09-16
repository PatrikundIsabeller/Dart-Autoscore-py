# post_homography.py
import requests, numpy as np

H = np.array(
    [
        [2.3, 0.0, 640.0],
        [0.0, -2.3, 360.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)
r = requests.post(
    "http://127.0.0.1:8017/set_homography", json={"camera_id": "cam0", "H": H.tolist()}
)
print(r.json())
