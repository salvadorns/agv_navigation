"""
fusion.py — Fusión multicámara con pesos adaptativos.

Combina las trayectorias de las 3 cámaras con la filosofía:
- Frontal = principal
- Izquierda = secundaria fuerte
- Derecha = soporte suave (pero puede ganar peso si detecta curva derecha)

La lógica de pesos es EXPLÍCITA y LEGIBLE. Cada ajuste tiene una razón clara.
"""

from collections import deque
from typing import Dict, Optional

import config
from nav_types import (
    CameraPosition, CameraResult, FusionResult
)


class MultiCameraFusion:
    """
    Fusiona información de las 3 cámaras en un resultado de navegación unificado.

    Estrategia de pesos adaptativos:
    1. Empezar con pesos base (F=0.6, L=0.3, R=0.1)
    2. Redistribuir si la cámara frontal tiene baja calidad
    3. Excluir cámaras con calidad ~0
    4. Aumentar peso de la derecha si detecta curva derecha sostenida
    5. Siempre renormalizar para que la suma sea 1.0
    """

    def __init__(self):
        # Historial del ahead_point de la cámara derecha para detección de curva
        self._right_curve_history = deque(
            maxlen=config.FUSION_RIGHT_CURVE_PERSISTENCE + 2
        )
        # Pesos efectivos del último frame (para debug)
        self.last_weights: Dict[str, float] = {}

    def fuse(
        self, camera_results: Dict[CameraPosition, CameraResult]
    ) -> FusionResult:
        """
        Fusiona los resultados de las 3 cámaras.

        Args:
            camera_results: Resultados del procesamiento de cada cámara.

        Returns:
            FusionResult con ahead point y heading fusionados.
        """
        # Extraer datos usables
        usable = {
            pos: result for pos, result in camera_results.items()
            if result.is_usable
        }

        num_valid = len(usable)

        if num_valid == 0:
            return FusionResult(
                ahead_point_norm=0.0,
                heading_angle=0.0,
                effective_weights={},
                global_quality=0.0,
                num_valid_cameras=0,
                valid=False
            )

        # --- Paso 1: Pesos base ---
        weights = {
            CameraPosition.FRONT: config.FUSION_WEIGHT_FRONT,
            CameraPosition.LEFT: config.FUSION_WEIGHT_LEFT,
            CameraPosition.RIGHT: config.FUSION_WEIGHT_RIGHT,
        }

        # --- Paso 2: Excluir cámaras no usables ---
        for pos in CameraPosition:
            if pos not in usable:
                weights[pos] = 0.0

        # --- Paso 3: Redistribuir si la frontal tiene baja calidad ---
        front_result = camera_results.get(CameraPosition.FRONT)
        if front_result and front_result.is_usable:
            front_quality = front_result.quality.total
            if front_quality < config.FUSION_FRONT_QUALITY_THRESHOLD:
                # Reducir peso de la frontal proporcionalmente a su calidad
                reduction = weights[CameraPosition.FRONT] * (
                    1.0 - front_quality / config.FUSION_FRONT_QUALITY_THRESHOLD
                )
                weights[CameraPosition.FRONT] -= reduction

                # Distribuir la reducción entre las cámaras laterales disponibles
                lateral_positions = [
                    p for p in [CameraPosition.LEFT, CameraPosition.RIGHT]
                    if p in usable
                ]
                if lateral_positions:
                    extra_per_cam = reduction / len(lateral_positions)
                    for p in lateral_positions:
                        weights[p] += extra_per_cam

        # --- Paso 4: Detección de curva derecha ---
        right_result = camera_results.get(CameraPosition.RIGHT)
        right_curve_detected = self._detect_right_curve(right_result)

        if right_curve_detected and CameraPosition.RIGHT in usable:
            # Aumentar peso de la derecha
            target_right = config.FUSION_RIGHT_CURVE_WEIGHT
            current_right = weights[CameraPosition.RIGHT]
            boost = target_right - current_right

            if boost > 0:
                weights[CameraPosition.RIGHT] = target_right
                # Reducir el extra de la frontal
                weights[CameraPosition.FRONT] = max(
                    0.1, weights[CameraPosition.FRONT] - boost
                )

        # --- Paso 5: Aplicar quality como factor multiplicativo ---
        # Los pesos se modulan por la calidad de cada cámara
        for pos in CameraPosition:
            if pos in usable:
                quality = camera_results[pos].quality.total
                # No multiplicar directamente, sino ponderar:
                # peso_efectivo = peso_base × (0.5 + 0.5 × quality)
                # Esto evita anular una cámara con calidad baja pero no cero
                weights[pos] *= (0.5 + 0.5 * quality)

        # --- Paso 6: Renormalizar ---
        total_weight = sum(weights.values())
        if total_weight > 0:
            for pos in weights:
                weights[pos] /= total_weight
        else:
            return FusionResult(valid=False, num_valid_cameras=0)

        # --- Paso 7: Fusionar ahead point y heading ---
        fused_ahead = 0.0
        fused_heading = 0.0

        for pos, result in usable.items():
            w = weights[pos]
            fused_ahead += w * result.trajectory.ahead_point_norm
            fused_heading += w * result.trajectory.heading_angle

        # --- Calidad global ponderada ---
        global_quality = sum(
            weights[pos] * camera_results[pos].quality.total
            for pos in usable
        )

        # Guardar pesos para debug
        self.last_weights = {pos.value: round(w, 3) for pos, w in weights.items()}

        return FusionResult(
            ahead_point_norm=fused_ahead,
            heading_angle=fused_heading,
            effective_weights=self.last_weights,
            global_quality=global_quality,
            num_valid_cameras=num_valid,
            valid=True
        )

    def _detect_right_curve(
        self, right_result: Optional[CameraResult]
    ) -> bool:
        """
        Detecta si la cámara derecha muestra una curva a la derecha sostenida.

        Criterio: el ahead point normalizado de la cámara derecha es
        consistentemente negativo (camino hacia la izquierda de su frame
        = curva a la derecha del AGV) durante N frames consecutivos.

        Returns:
            True si se detecta curva derecha sostenida.
        """
        if right_result is None or not right_result.is_usable:
            self._right_curve_history.append(0.0)
            return False

        ahead_norm = right_result.trajectory.ahead_point_norm
        self._right_curve_history.append(ahead_norm)

        # Verificar si hay suficientes frames y todos indican curva
        if len(self._right_curve_history) < config.FUSION_RIGHT_CURVE_PERSISTENCE:
            return False

        recent = list(self._right_curve_history)[
            -config.FUSION_RIGHT_CURVE_PERSISTENCE:
        ]

        # Todos los valores recientes deben indicar curva derecha
        all_curving = all(
            v < config.FUSION_RIGHT_CURVE_THRESHOLD for v in recent
        )

        return all_curving
