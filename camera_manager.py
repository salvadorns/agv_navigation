"""
camera_manager.py — Gestión de lectura de las 3 cámaras USB.

Maneja apertura, lectura sincronizada, y fallas de cámaras individuales.
"""

import time
from typing import Dict, Optional

import cv2
import numpy as np

import config
from nav_types import CameraFrame, CameraPosition


class CameraManager:
    """
    Administra las 3 cámaras USB del AGV.

    Responsabilidades:
    - Abrir y configurar cada cámara con resolución/FPS deseados
    - Leer frames de las 3 cámaras
    - Manejar fallas individuales sin crashear
    - Proveer detección automática de cámaras disponibles
    """

    def __init__(self, camera_indices: Optional[Dict[str, int]] = None):
        """
        Args:
            camera_indices: Mapeo {"left": idx, "front": idx, "right": idx}.
                            Si es None, usa config.CAMERA_INDICES.
        """
        self.indices = camera_indices or config.CAMERA_INDICES
        self.captures: Dict[CameraPosition, Optional[cv2.VideoCapture]] = {}
        self.last_valid_time: Dict[CameraPosition, float] = {}
        self._open_cameras()

    def _open_cameras(self):
        """Abre todas las cámaras configuradas."""
        position_map = {
            "left": CameraPosition.LEFT,
            "front": CameraPosition.FRONT,
            "right": CameraPosition.RIGHT,
        }

        for name, index in self.indices.items():
            pos = position_map[name]
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW para Windows

            if cap.isOpened():
                # Configurar resolución y FPS
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
                # Reducir buffer para menor latencia
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                self.captures[pos] = cap
                self.last_valid_time[pos] = time.time()
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[CameraManager] {pos.value} (idx={index}): "
                      f"abierta OK — {actual_w}x{actual_h}")
            else:
                self.captures[pos] = None
                self.last_valid_time[pos] = 0
                print(f"[CameraManager] {pos.value} (idx={index}): "
                      f"NO se pudo abrir")

    def read_all(self) -> Dict[CameraPosition, CameraFrame]:
        """
        Lee un frame de cada cámara.

        Returns:
            Dict con CameraFrame por posición. Si una cámara falla,
            el CameraFrame tendrá valid=False y frame=None.
        """
        frames: Dict[CameraPosition, CameraFrame] = {}
        now = time.time()

        for pos in CameraPosition:
            cap = self.captures.get(pos)

            if cap is None or not cap.isOpened():
                frames[pos] = CameraFrame(
                    frame=None,
                    camera_id=pos,
                    timestamp=now,
                    valid=False
                )
                continue

            ret, frame = cap.read()

            if ret and frame is not None and frame.size > 0:
                self.last_valid_time[pos] = now
                frames[pos] = CameraFrame(
                    frame=frame,
                    camera_id=pos,
                    timestamp=now,
                    valid=True
                )
            else:
                # Verificar si supera el timeout
                elapsed = now - self.last_valid_time.get(pos, 0)
                if elapsed > config.CAMERA_TIMEOUT_SEC:
                    print(f"[CameraManager] WARN: {pos.value} timeout "
                          f"({elapsed:.1f}s sin frame válido)")
                frames[pos] = CameraFrame(
                    frame=None,
                    camera_id=pos,
                    timestamp=now,
                    valid=False
                )

        return frames

    def get_status(self) -> Dict[str, bool]:
        """Devuelve el estado de conexión de cada cámara."""
        return {
            pos.value: (cap is not None and cap.isOpened())
            for pos, cap in self.captures.items()
        }

    def release_all(self):
        """Libera todos los recursos de cámara."""
        for pos, cap in self.captures.items():
            if cap is not None:
                cap.release()
                print(f"[CameraManager] {pos.value}: liberada")
        self.captures.clear()


def detect_available_cameras(max_index: int = 10) -> list:
    """
    Detecta cámaras disponibles probando índices secuenciales.

    Útil para descubrir qué índices están activos antes de configurar.

    Args:
        max_index: Máximo índice a probar.

    Returns:
        Lista de índices de cámaras detectadas.
    """
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available.append(i)
            cap.release()
    return available
