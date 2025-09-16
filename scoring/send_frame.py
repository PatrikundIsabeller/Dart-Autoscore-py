# send_frame.py
import cv2 as cv, base64, requests

CAMERA_ID = "cam0"
cap = cv.VideoCapture(0)
ok, frame = cap.read()
cap.release()
assert ok, "Kamera gibt kein Bild"
ok, enc = cv.imencode(".jpg", frame)
assert ok
img_b64 = base64.b64encode(enc.tobytes()).decode("ascii")
r = requests.post(
    "http://127.0.0.1:8017/detect",
    json={"camera_id": CAMERA_ID, "image_base64": img_b64},
)
print(r.json())
