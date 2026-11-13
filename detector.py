"""
detector.py — Detección de objetos y segmentación auxiliar.

Corre en paralelo con el pipeline principal:
  - yolov8n.pt  → bounding boxes de personas, autos, motos, buses, camiones
  - 2cams.pt    → segmentación auxiliar (overlay naranja)
"""

import os
import numpy as np
import cv2
from ultralytics import YOLO

YOLOV8N_PATH = r"c:\Users\Novillo\OneDrive\Desktop\autonomouse vehicle\yolov8n.pt"
TWOCAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2cams.pt")

# Clases COCO de interés: id → (nombre, color BGR)
COCO_CLASSES = {
    0:  ("Persona",   (0,   0,   255)),
    2:  ("Auto",      (255, 0,   0  )),
    3:  ("Moto",      (255, 128, 0  )),
    5:  ("Bus",       (128, 0,   255)),
    7:  ("Camion",    (0,   128, 255)),
}

# Color del overlay de segmentación de 2cams
TWOCAMS_OVERLAY_COLOR = (0, 165, 255)   # Naranja
TWOCAMS_OVERLAY_ALPHA = 0.4


class DetectionOutput:
    def __init__(self):
        self.boxes   = []          # list of (label, conf, (x1,y1,x2,y2), color)
        self.seg_overlay = None    # np.ndarray o None


class MultiDetector:
    def __init__(self):
        print("[Detector] Cargando yolov8n (deteccion de objetos)...")
        self.detector  = YOLO(YOLOV8N_PATH)

        print("[Detector] Cargando 2cams (segmentacion auxiliar)...")
        self.segmentor = YOLO(TWOCAMS_PATH)

        print("[Detector] Modelos listos.")

    def run(self, frame: np.ndarray) -> DetectionOutput:
        out = DetectionOutput()
        if frame is None:
            return out

        # --- Deteccion de objetos ---
        det_res = self.detector(frame, verbose=False)[0]
        for box in det_res.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in COCO_CLASSES:
                continue
            label, color = COCO_CLASSES[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            out.boxes.append((label, conf, (x1, y1, x2, y2), color))

        # --- Segmentacion 2cams ---
        seg_res = self.segmentor(frame, verbose=False)[0]
        h, w = frame.shape[:2]

        if seg_res.masks is not None:
            overlay = np.zeros_like(frame)
            for mask_tensor in seg_res.masks.data:
                mask_np = mask_tensor.cpu().numpy()
                mask_rs = cv2.resize(mask_np, (w, h))
                overlay[mask_rs > 0.5] = TWOCAMS_OVERLAY_COLOR
            out.seg_overlay = overlay
        elif seg_res.boxes is not None and len(seg_res.boxes):
            # 2cams es modelo de deteccion, no de segmentacion
            for box in seg_res.boxes:
                cls_id = int(box.cls[0])
                label = seg_res.names.get(cls_id, str(cls_id))
                conf  = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                out.boxes.append((label, conf, (x1, y1, x2, y2), TWOCAMS_OVERLAY_COLOR))

        return out
