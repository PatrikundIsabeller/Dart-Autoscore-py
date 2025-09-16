# post_homography_from_json.py — setzt H für cam0/cam1/cam2 aus kalibrierung.json
import json, requests, numpy as np
from pathlib import Path

API = "http://127.0.0.1:8017"
JSON_PATH = Path(
    r"C:\Users\PatrikPesendorfer\dart-autoscore-py\kalibrierung.json"
)  # ggf. anpassen


def find_H(data, cam_id):
    """
    Versucht mehrere gängige Strukturen:
    - data[cam_id]["H_world2img"] oder ["H"]
    - data["homographies"][cam_id], …
    Passe das Mapping ggf. an deine Datei an.
    """
    if cam_id in data:
        node = data[cam_id]
        for k in ("H_world2img", "H", "world2image", "homography"):
            if k in node:
                return node[k]
    for key in ("homographies", "cameras", "cams"):
        if key in data and cam_id in data[key]:
            node = data[key][cam_id]
            for k in ("H_world2img", "H", "world2image", "homography"):
                if k in node:
                    return node[k]
    return None


def post_H(cam, H):
    H = np.array(H, dtype=float).reshape(3, 3)
    r = requests.post(f"{API}/set_homography", json={"camera_id": cam, "H": H.tolist()})
    print(cam, r.json())


def main():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    for cam in ("cam0", "cam1", "cam2"):
        H = find_H(data, cam)
        if H is None:
            print(f"{cam}: keine Homography im JSON gefunden – bitte Mapping anpassen.")
        else:
            post_H(cam, H)


if __name__ == "__main__":
    main()
