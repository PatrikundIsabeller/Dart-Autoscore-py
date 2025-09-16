# client_example.py
import base64, json, cv2 as cv, numpy as np, requests
API = "http://127.0.0.1:8017"; CAM = "cam0"
H = np.array([[2.3, 0.0, 640.0],[0.0, -2.3, 360.0],[0.0, 0.0, 1.0]], dtype=np.float64)
r = requests.post(f"{API}/set_homography", json={"camera_id": CAM, "H": H.tolist()})
print("set_homography:", r.json())
img = cv.imread("sample_frame.jpg"); _, enc = cv.imencode(".jpg", img)
img_b64 = base64.b64encode(enc.tobytes()).decode("ascii")
r = requests.post(f"{API}/detect", json={"camera_id": CAM, "image_base64": img_b64})
print(json.dumps(r.json(), indent=2))
