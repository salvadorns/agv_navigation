"""
trajectory.py — Extracción de trayectoria del camino desde la máscara.

Analiza la máscara limpia por franjas horizontales para construir
una centerline y estimar un ahead point de navegación.
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from nav_types import CenterlinePoint, TrajectoryResult


class TrajectoryExtractor:
    """
    Extrae la trayectoria central del camino a partir de una máscara binaria.

    Proceso:
    1. Definir ROI (ignorar parte superior = cielo/horizonte)
    2. Dividir en franjas horizontales
    3. En cada franja: detectar bordes izq/der del camino → centro
    4. Construir centerline
    5. Estimar ahead point
    6. Estimar heading local
    """

    def __init__(self):
        self.img_width = config.CAMERA_WIDTH
        self.img_height = config.CAMERA_HEIGHT
        self.num_slices = config.TRAJECTORY_NUM_SLICES
        self.top_ignore = config.TRAJECTORY_TOP_IGNORE_RATIO
        self.lookahead = config.TRAJECTORY_LOOKAHEAD_RATIO
        self.min_width = config.TRAJECTORY_MIN_PATH_WIDTH

    def extract(self, mask: np.ndarray) -> TrajectoryResult:
        """
        Extrae trayectoria de una máscara de segmentación.

        Args:
            mask: Máscara binaria uint8 [0,255] del camino.

        Returns:
            TrajectoryResult con centerline, ahead point, heading y metadata.
        """
        if mask is None or mask.size == 0 or np.sum(mask) == 0:
            return TrajectoryResult(
                centerline=[], ahead_point=None,
                ahead_point_norm=0.0, heading_angle=0.0,
                valid=False, num_valid_slices=0
            )

        h, w = mask.shape[:2]

        # 1. Definir ROI vertical (ignorar parte superior)
        roi_top = int(h * self.top_ignore)
        roi_bottom = h
        roi_height = roi_bottom - roi_top

        if roi_height < self.num_slices:
            return TrajectoryResult(
                centerline=[], ahead_point=None,
                valid=False, num_valid_slices=0
            )

        # 2. Calcular posiciones de las franjas (de abajo hacia arriba)
        slice_height = roi_height / self.num_slices
        centerline: List[CenterlinePoint] = []

        for i in range(self.num_slices):
            # Franja i: de abajo (i=0) a arriba (i=num_slices-1)
            y_center = int(roi_bottom - (i + 0.5) * slice_height)
            y_top = int(roi_bottom - (i + 1) * slice_height)
            y_bot = int(roi_bottom - i * slice_height)

            # Asegurar que estamos dentro de límites
            y_top = max(y_top, roi_top)
            y_bot = min(y_bot, roi_bottom)
            y_center = max(roi_top, min(y_center, roi_bottom - 1))

            # 3. Analizar la franja: promedio de las filas en la franja
            stripe = mask[y_top:y_bot, :]

            if stripe.size == 0:
                continue

            # Proyectar verticalmente: para cada columna, si hay máscara
            col_presence = np.any(stripe > 127, axis=0)
            active_cols = np.where(col_presence)[0]

            if len(active_cols) < self.min_width:
                continue

            # Borde izquierdo y derecho del camino en esta franja
            left_edge = active_cols[0]
            right_edge = active_cols[-1]
            path_width = right_edge - left_edge

            if path_width < self.min_width:
                continue

            # Centro del camino
            center_x = (left_edge + right_edge) / 2.0

            centerline.append(CenterlinePoint(
                x=center_x,
                y=float(y_center),
                width=float(path_width)
            ))

        num_valid = len(centerline)

        if num_valid < 2:
            return TrajectoryResult(
                centerline=centerline, ahead_point=None,
                valid=False, num_valid_slices=num_valid
            )

        # 4. Estimar ahead point
        ahead_point, ahead_norm = self._compute_ahead_point(centerline, w)

        # 5. Estimar heading
        heading = self._compute_heading(centerline)

        return TrajectoryResult(
            centerline=centerline,
            ahead_point=ahead_point,
            ahead_point_norm=ahead_norm,
            heading_angle=heading,
            valid=True,
            num_valid_slices=num_valid
        )

    def _compute_ahead_point(
        self, centerline: List[CenterlinePoint], img_width: int
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Calcula el ahead point interpolando en la centerline.

        El ahead point está a una fracción `lookahead_ratio` de altura
        desde la base de la centerline.

        Returns:
            (ahead_point_pixels, ahead_point_normalized)
        """
        if not centerline:
            return None, 0.0

        # La centerline va de abajo (idx 0) a arriba (idx -1)
        # Queremos el punto a `lookahead` fracción de la lista
        target_idx = self.lookahead * (len(centerline) - 1)
        idx_low = int(target_idx)
        idx_high = min(idx_low + 1, len(centerline) - 1)
        frac = target_idx - idx_low

        # Interpolación lineal
        pt_low = centerline[idx_low]
        pt_high = centerline[idx_high]
        ahead_x = pt_low.x + frac * (pt_high.x - pt_low.x)
        ahead_y = pt_low.y + frac * (pt_high.y - pt_low.y)

        # Normalizar horizontalmente: centro de imagen = 0, izquierda = -1, derecha = +1
        center_x = img_width / 2.0
        ahead_norm = (ahead_x - center_x) / center_x  # [-1, 1]
        ahead_norm = max(-1.0, min(1.0, ahead_norm))   # Clamp

        return (ahead_x, ahead_y), ahead_norm

    def _compute_heading(self, centerline: List[CenterlinePoint]) -> float:
        """
        Estima el ángulo de heading del camino usando regresión lineal
        sobre los puntos superiores de la centerline.

        Returns:
            Ángulo en grados. 0 = recto, positivo = camino va a la derecha,
            negativo = camino va a la izquierda.
        """
        if len(centerline) < 3:
            return 0.0

        # Usar la mitad superior de la centerline para estimar dirección
        n_upper = max(3, len(centerline) // 2)
        upper_points = centerline[-n_upper:]

        xs = np.array([p.x for p in upper_points])
        ys = np.array([p.y for p in upper_points])

        # Regresión lineal: x = m*y + b
        # (Usamos y como variable independiente porque la centerline es más vertical)
        if np.std(ys) < 1.0:
            return 0.0

        # Ajuste por mínimos cuadrados
        coeffs = np.polyfit(ys, xs, 1)
        slope = coeffs[0]  # dx/dy: cuánto se mueve horizontalmente por píxel vertical

        # Convertir a ángulo (atan da radianes, convertir a grados)
        # slope negativo = camino va a la izquierda cuando subimos (y decrece)
        # Invertir porque y crece hacia abajo en la imagen
        angle_rad = math.atan(-slope)
        angle_deg = math.degrees(angle_rad)

        # Clamp a rango razonable
        return max(-60.0, min(60.0, angle_deg))
