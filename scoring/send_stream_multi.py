# send_stream_multi.py — streamt 3 Kameras parallel an /detect (STRG+C zum Beenden)
import time, base64, threading, requests, cv2 as cv

API = "http://127.0.0.1:8017"
CAM_MAP = {  # Kamera-IDs -> VideoCapture-Index anpassen falls nötig
    "cam0": 0,
    "cam1": 1,
    "cam2": 2,
}
INTERVAL_S = 0.30  # ~300 ms pro Kamera
JPEG_QUALITY = 80


def worker(cam_id, cap_index):
    cap = cv.VideoCapture(cap_index, cv.CAP_DSHOW)  # CAP_DSHOW hilft unter Windows
    if not cap.isOpened():
        print(f"[{cam_id}] Kamera {cap_index} nicht geöffnet")
        return
    print(f"[{cam_id}] Stream gestartet (Index {cap_index})")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(INTERVAL_S)
                continue
            ok, enc = cv.imencode(
                ".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not ok:
                time.sleep(INTERVAL_S)
                continue
            b64 = base64.b64encode(enc.tobytes()).decode("ascii")
            try:
                r = requests.post(
                    f"{API}/detect",
                    json={"camera_id": cam_id, "image_base64": b64},
                    timeout=2,
                )
                data = r.json()
                if data.get("hits"):
                    print(f"[{cam_id}] Hits: {data['hits']}")
            except Exception as e:
                print(f"[{cam_id}] POST /detect error:", e)
            time.sleep(INTERVAL_S)
    finally:
        cap.release()
        print(f"[{cam_id}] Stream beendet")


def main():
    threads = []
    for cam_id, idx in CAM_MAP.items():
        t = threading.Thread(target=worker, args=(cam_id, idx), daemon=True)
        t.start()
        threads.append(t)
    print("Streams laufen – STRG+C zum Stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stop…")


if __name__ == "__main__":
    main()
