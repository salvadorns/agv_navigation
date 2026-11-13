"""
inference.py — Inferencia de segmentación con YOLOv8.

Carga el modelo una sola vez y ejecuta predicción por frame.
Selecciona la máscara principal del camino con regla robusta.
"""

from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

import config
from nav_types import MaskResult


class SegmentationInference:
    """
    Ejecuta inferencia de segmentación YOLOv8.

    El modelo se carga una sola vez en el constructor.
    Cada llamada a run() procesa un frame y devuelve la máscara principal.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Ruta al archivo .pt del modelo. Default: config.MODEL_PATH.
        """
        path = model_path or config.MODEL_PATH
        print(f"[Inference] Cargando modelo: {path}")

        # Detectar device automáticamente si no está configurado
        device = config.INFERENCE_DEVICE
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(path)
        self.device = device
        print(f"[Inference] Modelo cargado. Device: {self.device}")

    def run(self, frame: np.ndarray) -> MaskResult:
        """
        Ejecuta inferencia en un frame y devuelve la máscara principal del camino.

        Estrategia de selección de máscara:
        1. Ejecutar predicción YOLO
        2. Si hay filtro de clase, aplicarlo
        3. De todas las máscaras, seleccionar la que maximice: area × confianza
        4. Redimensionar la máscara al tamaño original del frame
        5. Si ninguna máscara supera los umbrales → MaskResult(valid=False)

        Args:
            frame: Imagen BGR de la cámara.

        Returns:
            MaskResult con la máscara binaria y metadata.
        """
        if frame is None:
            return MaskResult(mask=None, confidence=0.0, valid=False)

        h_orig, w_orig = frame.shape[:2]

        # Ejecutar inferencia
        results = self.model.predict(
            source=frame,
            imgsz=config.INFERENCE_IMGSZ,
            conf=config.INFERENCE_CONF_THRESHOLD,
            device=self.device,
            verbose=False,
            retina_masks=True,  # Máscaras de mayor resolución
        )

        # Verificar que hay resultados con máscaras
        if (not results or
            results[0].masks is None or
            len(results[0].masks.data) == 0):
            return MaskResult(mask=None, confidence=0.0, valid=False)

        result = results[0]
        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()           # (N,)
        class_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)
        
        # Las máscaras en formato tensor (masks.data) contienen el padding
        # del letterboxing. Si las redimensionamos directamente, la máscara
        # queda aplastada (tu corte a la mitad).
        # Solución: usar result.masks.xy, que son los polígonos ya escalados
        # a las dimensiones originales reales de la imagen por YOLO.
        polygons = result.masks.xy

        # Filtrar por clase de interés si está configurado
        valid_idx = list(range(len(class_ids)))
        if config.TARGET_CLASS_ID is not None:
            valid_idx = [i for i, cid in enumerate(class_ids)
                         if cid == config.TARGET_CLASS_ID]
            if not valid_idx:
                return MaskResult(mask=None, confidence=0.0, valid=False)

        # FUSIONAR todas las máscaras válidas dibujando sus polígonos
        combined_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        max_confidence = 0.0
        dominant_class = int(class_ids[valid_idx[0]])
        total_area = 0
        weighted_conf = 0.0

        for i in valid_idx:
            # Dibujar polígono en coordenadas originales
            poly = polygons[i]
            if len(poly) > 0:
                pts = np.array(poly, dtype=np.int32)
                # Dibujar máscara temp para calcular su área exacta a posteriori si quisieramos, 
                # pero para evitar sobreponer áreas, podemos simplemente dibujar en la combinada:
                cv2.fillPoly(combined_mask, [pts], 255)
                
            if confidences[i] > max_confidence:
                max_confidence = float(confidences[i])
                dominant_class = int(class_ids[i])
                
            # Aproximación del área basada en bounding box o asumida igual para pesos
            area_w = boxes.xywh[i][2].item() * boxes.xywh[i][3].item()
            weighted_conf += area_w * confidences[i]
            total_area += area_w

        avg_confidence = weighted_conf / max(total_area, 1)

        # Verificar que la máscara combinada tiene área útil
        if np.sum(combined_mask > 0) < config.MASK_MIN_BLOB_AREA:
            return MaskResult(mask=None, confidence=0.0, valid=False)

        return MaskResult(
            mask=combined_mask,
            confidence=float(avg_confidence),
            valid=True,
            class_id=dominant_class
        )
