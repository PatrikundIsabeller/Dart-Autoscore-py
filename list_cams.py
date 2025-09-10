# list_cams.py
import cv2


def open_capture(cam_id, res=(320, 240), fps=15):
    backends = [
        getattr(cv2, "CAP_MSMF", 1400),
        getattr(cv2, "CAP_DSHOW", 700),
        getattr(cv2, "CAP_ANY", 0),
    ]
    for be in backends:
        cap = cv2.VideoCapture(cam_id, be)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        if cap.isOpened():
            cap.release()
            return True, be
        cap.release()
    return False, None


print("Suche Kameras 0..10 (MSMF→DSHOW→ANY):")
for i in range(11):
    ok, be = open_capture(i)
    if ok:
        print(f"  ✔ Cam {i} OK (backend={be})")
    else:
        print(f"  ✖ Cam {i} nicht verfügbar")
