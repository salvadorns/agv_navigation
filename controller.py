"""
controller.py — Controlador de navegación del AGV.

Convierte el ahead point fusionado y heading en comandos
de navegación (linear.x, angular.z) con suavizado temporal
y políticas de seguridad.
"""

import config
from nav_types import FusionResult, NavCommand, SystemState


class NavigationController:
    """
    Controlador proporcional para navegación del AGV.

    A partir del error lateral (ahead point) y error angular (heading),
    genera comandos de velocidad lineal y angular con:
    - Reducción de velocidad en curvas
    - Suavizado temporal (EMA) para evitar jitter
    - Políticas de seguridad (parar si calidad baja, etc.)
    """

    def __init__(self):
        # Estado suavizado (EMA)
        self._smooth_linear = 0.0
        self._smooth_angular = 0.0
        # Último estado del sistema
        self.last_state = SystemState.STOPPED

    def compute(self, fusion: FusionResult) -> NavCommand:
        """
        Calcula el comando de navegación a partir de la fusión multicámara.

        Pipeline:
        1. Verificar seguridad / fallbacks
        2. Calcular errores (lateral y angular)
        3. Calcular angular.z y linear.x raw
        4. Aplicar suavizado temporal
        5. Aplicar límites

        Args:
            fusion: Resultado de la fusión multicámara.

        Returns:
            NavCommand con linear_x, angular_z y estado.
        """

        # =====================================================================
        # 1. SEGURIDAD Y FALLBACKS
        # =====================================================================

        # Sin cámaras válidas → parada total inmediata
        if not fusion.valid or fusion.num_valid_cameras == 0:
            return self._stop(SystemState.NO_CAMERAS, fusion.global_quality)

        # Calidad global crítica → parada de seguridad
        if fusion.global_quality < config.SAFETY_STOP_THRESHOLD:
            return self._stop(SystemState.STOPPED, fusion.global_quality)

        # Pocas cámaras válidas
        if fusion.num_valid_cameras < config.SAFETY_MIN_VALID_CAMERAS:
            return self._stop(SystemState.STOPPED, fusion.global_quality)

        # =====================================================================
        # 2. CALCULAR ERRORES
        # =====================================================================

        # Error lateral: desviación horizontal del ahead point respecto al centro
        # ahead_point_norm > 0 → camino a la derecha → girar a la derecha (angular.z negativo)
        # ahead_point_norm < 0 → camino a la izquierda → girar a la izquierda (angular.z positivo)
        error_lateral = fusion.ahead_point_norm

        # Error angular: ángulo del heading del camino
        # heading > 0 → camino se curva a la derecha → anticipar giro
        # Normalizar a [-1, 1] dividiendo por un rango razonable (e.g., 45 grados)
        error_angular = fusion.heading_angle / 45.0
        error_angular = max(-1.0, min(1.0, error_angular))

        # =====================================================================
        # 3. CALCULAR COMANDOS RAW
        # =====================================================================

        # angular.z: combinación de error lateral y angular
        # Signo negativo porque: error positivo (camino a la derecha) → giro derecho
        raw_angular = -(
            config.CONTROL_KP_LATERAL * error_lateral +
            config.CONTROL_KP_ANGULAR * error_angular
        )

        # Limitar angular.z
        raw_angular = max(-config.CONTROL_MAX_ANGULAR_Z,
                          min(config.CONTROL_MAX_ANGULAR_Z, raw_angular))

        # linear.x: reducir velocidad proporcional al giro
        # Más giro → menos velocidad lineal
        turn_magnitude = abs(raw_angular)
        raw_linear = config.CONTROL_MAX_LINEAR_X * (
            1.0 - config.CONTROL_TURN_REDUCTION_FACTOR * turn_magnitude
        )

        # Asegurar velocidad mínima (no detenerse en curvas)
        raw_linear = max(config.CONTROL_MIN_LINEAR_X, raw_linear)

        # =====================================================================
        # 4. DETERMINAR ESTADO
        # =====================================================================

        state = SystemState.NORMAL

        if fusion.global_quality < config.SAFETY_LOW_CONFIDENCE_THRESHOLD:
            state = SystemState.LOW_CONFIDENCE
            # Reducir velocidad al 50% en baja confianza
            raw_linear *= 0.5

        # =====================================================================
        # 5. SUAVIZADO TEMPORAL (EMA)
        # =====================================================================

        alpha = config.CONTROL_SMOOTHING_ALPHA
        self._smooth_linear = alpha * raw_linear + (1 - alpha) * self._smooth_linear
        self._smooth_angular = alpha * raw_angular + (1 - alpha) * self._smooth_angular

        self.last_state = state

        return NavCommand(
            linear_x=round(self._smooth_linear, 4),
            angular_z=round(self._smooth_angular, 4),
            state=state,
            global_quality=fusion.global_quality
        )

    def _stop(self, state: SystemState, quality: float) -> NavCommand:
        """
        Genera un comando de parada, con decaimiento suave de la velocidad.

        No para bruscamente, sino que reduce rápidamente a 0 usando EMA
        con alpha alto para convergencia rápida.
        """
        # Decaimiento rápido hacia 0
        self._smooth_linear *= 0.3   # Reduce rápido
        self._smooth_angular *= 0.3

        # Si ya está casi en 0, fijar a 0
        if abs(self._smooth_linear) < 0.01:
            self._smooth_linear = 0.0
        if abs(self._smooth_angular) < 0.01:
            self._smooth_angular = 0.0

        self.last_state = state

        return NavCommand(
            linear_x=round(self._smooth_linear, 4),
            angular_z=round(self._smooth_angular, 4),
            state=state,
            global_quality=quality
        )

    def reset(self):
        """Resetea el estado interno del controlador."""
        self._smooth_linear = 0.0
        self._smooth_angular = 0.0
        self.last_state = SystemState.STOPPED
