"""
quality.py — Evaluación de calidad por cámara.

Score compuesto [0,1] que combina múltiples factores para determinar
qué tan confiable es la información de cada cámara en el frame actual.
"""

from collections import deque
from typing import Dict, Optional

import numpy as np

import config
from nav_types import (
    CameraPosition, MaskResult, TrajectoryResult, QualityScore
)


class QualityEvaluator:
    """
    Evalúa la calidad de la información de cada cámara.

    Factores evaluados:
    - Área de máscara útil (¿hay suficiente camino visible?)
    - Continuidad vertical (¿el camino es continuo de abajo a arriba?)
    - Confianza de YOLO (¿qué tan seguro está el modelo?)
    - Consistencia de la centerline (¿los puntos forman una trayectoria coherente?)
    - Estabilidad temporal (¿el ahead point es estable entre frames?)

    Mantiene historial temporal por cámara para el factor de estabilidad.
    """

    def __init__(self):
        # Historial del ahead_point normalizado por cámara
        self._history: Dict[CameraPosition, deque] = {
            pos: deque(maxlen=config.QUALITY_TEMPORAL_WINDOW)
            for pos in CameraPosition
        }

    def compute(
        self,
        camera_id: CameraPosition,
        mask: MaskResult,
        trajectory: TrajectoryResult,
        clean_mask: Optional[np.ndarray] = None
    ) -> QualityScore:
        """
        Calcula el score de calidad para una cámara.

        Args:
            camera_id: Posición de la cámara.
            mask: Resultado de la inferencia.
            trajectory: Resultado de la extracción de trayectoria.
            clean_mask: Máscara postprocesada (para cálculo de área).

        Returns:
            QualityScore con score total y desglose por factor.
        """
        if not mask.valid or not trajectory.valid:
            # Agregar 0 al historial para que la estabilidad refleje la falla
            self._history[camera_id].append(None)
            return QualityScore(total=0.0)

        # --- Factor 1: Área de máscara ---
        area_score = self._compute_area_score(clean_mask)

        # --- Factor 2: Continuidad vertical ---
        continuity_score = self._compute_continuity_score(trajectory)

        # --- Factor 3: Confianza YOLO ---
        confidence_score = min(1.0, mask.confidence)

        # --- Factor 4: Consistencia de la centerline ---
        consistency_score = self._compute_consistency_score(trajectory)

        # --- Factor 5: Estabilidad temporal ---
        stability_score = self._compute_stability_score(
            camera_id, trajectory.ahead_point_norm
        )

        # Score compuesto ponderado
        total = (
            config.QUALITY_WEIGHT_AREA * area_score +
            config.QUALITY_WEIGHT_CONTINUITY * continuity_score +
            config.QUALITY_WEIGHT_CONFIDENCE * confidence_score +
            config.QUALITY_WEIGHT_CONSISTENCY * consistency_score +
            config.QUALITY_WEIGHT_STABILITY * stability_score
        )

        # Clamp a [0, 1]
        total = max(0.0, min(1.0, total))

        return QualityScore(
            total=total,
            area_score=area_score,
            continuity_score=continuity_score,
            confidence_score=confidence_score,
            consistency_score=consistency_score,
            stability_score=stability_score
        )

    def _compute_area_score(self, mask: Optional[np.ndarray]) -> float:
        """
        Evalúa qué fracción del ROI está cubierta por la máscara.

        Un camino debería cubrir una fracción significativa del ROI.
        Muy poca área = detección pobre. Se satura al max esperado.
        """
        if mask is None or mask.size == 0:
            return 0.0

        h, w = mask.shape[:2]
        roi_top = int(h * config.TRAJECTORY_TOP_IGNORE_RATIO)
        roi_mask = mask[roi_top:, :]

        if roi_mask.size == 0:
            return 0.0

        total_pixels = roi_mask.shape[0] * roi_mask.shape[1]
        mask_pixels = np.sum(roi_mask > 127)

        ratio = mask_pixels / total_pixels
        # Normalizar respecto al máximo esperado
        score = min(1.0, ratio / config.QUALITY_MAX_AREA_RATIO)

        return score

    def _compute_continuity_score(self, trajectory: TrajectoryResult) -> float:
        """
        Evalúa qué fracción de las franjas tienen camino detectado.

        Un camino continuo debería aparecer en la mayoría de las franjas.
        """
        if config.TRAJECTORY_NUM_SLICES == 0:
            return 0.0
        return trajectory.num_valid_slices / config.TRAJECTORY_NUM_SLICES

    def _compute_consistency_score(self, trajectory: TrajectoryResult) -> float:
        """
        Evalúa qué tan coherente es la forma de la centerline.

        Una centerline con mucha variación lateral (zigzag) indica
        detección ruidosa. Medimos la desviación estándar normalizada
        de las posiciones x de la centerline.
        """
        if len(trajectory.centerline) < 3:
            return 0.5  # Score neutro si hay pocos puntos

        xs = [p.x for p in trajectory.centerline]
        mean_x = np.mean(xs)
        std_x = np.std(xs)

        # Normalizar por el ancho promedio del camino
        avg_width = np.mean([p.width for p in trajectory.centerline])
        if avg_width < 1.0:
            return 0.0

        # Desviación relativa al ancho del camino
        # Si std_x es pequeña respecto al ancho → alta consistencia
        relative_dev = std_x / avg_width
        score = max(0.0, 1.0 - relative_dev)

        return score

    def _compute_stability_score(
        self, camera_id: CameraPosition, ahead_norm: float
    ) -> float:
        """
        Evalúa qué tan estable es el ahead point entre frames consecutivos.

        Un ahead point que salta mucho indica detección inestable.
        """
        history = self._history[camera_id]

        # Agregar valor actual al historial
        history.append(ahead_norm)

        # Necesitamos al menos 2 valores para comparar
        valid_values = [v for v in history if v is not None]
        if len(valid_values) < 2:
            return 0.8  # Score optimista al inicio

        # Calcular variación entre frames consecutivos
        diffs = [abs(valid_values[i] - valid_values[i-1])
                 for i in range(1, len(valid_values))]
        mean_diff = np.mean(diffs)

        # Normalizar: diff de 0 = perfecto, diff de 0.5 = muy inestable
        score = max(0.0, 1.0 - mean_diff * 2.0)

        return score

    def reset(self):
        """Limpia el historial temporal de todas las cámaras."""
        for pos in CameraPosition:
            self._history[pos].clear()
