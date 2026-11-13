"""
nav_types.py — Tipos de datos del pipeline de navegación.

Dataclasses para intercambio limpio de información entre módulos.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import time

import numpy as np


class CameraPosition(Enum):
    """Posición de la cámara en el AGV."""
    LEFT = "left"
    FRONT = "front"
    RIGHT = "right"


class SystemState(Enum):
    """Estado operativo del sistema de navegación."""
    NORMAL = "NORMAL"                    # Operación normal
    LOW_CONFIDENCE = "LOW_CONFIDENCE"    # Baja confianza, velocidad reducida
    STOPPED = "STOPPED"                  # Detenido por seguridad
    NO_CAMERAS = "NO_CAMERAS"            # Sin cámaras disponibles


@dataclass
class CameraFrame:
    """Frame capturado de una cámara."""
    frame: Optional[np.ndarray]  # Imagen BGR, None si falla
    camera_id: CameraPosition
    timestamp: float = field(default_factory=time.time)
    valid: bool = True

    def __post_init__(self):
        if self.frame is None:
            self.valid = False


@dataclass
class MaskResult:
    """Resultado de la inferencia de segmentación para una cámara."""
    mask: Optional[np.ndarray] = None  # Máscara binaria uint8 [0,255], None si inválida
    confidence: float = 0.0          # Confianza de la detección [0,1]
    valid: bool = False              # Si se encontró una máscara usable
    class_id: int = -1               # ID de clase detectada


@dataclass
class CenterlinePoint:
    """Un punto de la centerline del camino."""
    x: float          # Posición horizontal (píxeles)
    y: float          # Posición vertical (píxeles)
    width: float      # Ancho del camino en esta franja (píxeles)


@dataclass
class TrajectoryResult:
    """Resultado de la extracción de trayectoria para una cámara."""
    centerline: List[CenterlinePoint]           # Puntos de la centerline (de abajo a arriba)
    ahead_point: Optional[Tuple[float, float]]  # Punto de lookahead (x, y) en píxeles
    ahead_point_norm: float = 0.0               # Ahead point normalizado [-1, 1] (horizontal)
    heading_angle: float = 0.0                  # Ángulo del camino en grados (-90 a 90)
    valid: bool = False                         # Si la trayectoria es usable
    num_valid_slices: int = 0                   # Número de franjas válidas


@dataclass
class QualityScore:
    """Score de calidad de una cámara."""
    total: float = 0.0              # Score compuesto [0, 1]
    area_score: float = 0.0        # Componente de área
    continuity_score: float = 0.0  # Componente de continuidad
    confidence_score: float = 0.0  # Componente de confianza
    consistency_score: float = 0.0 # Componente de consistencia geométrica
    stability_score: float = 0.0   # Componente de estabilidad temporal


@dataclass
class CameraResult:
    """Resultado completo del procesamiento de una cámara."""
    camera_id: CameraPosition
    frame: Optional[np.ndarray]
    mask: MaskResult
    clean_mask: Optional[np.ndarray]   # Máscara postprocesada
    trajectory: TrajectoryResult
    quality: QualityScore

    @property
    def is_usable(self) -> bool:
        """Si esta cámara contribuye datos usables."""
        return self.mask.valid and self.trajectory.valid and self.quality.total > 0.05


@dataclass
class FusionResult:
    """Resultado de la fusión multicámara."""
    ahead_point_norm: float = 0.0            # Ahead point fusionado normalizado [-1, 1]
    heading_angle: float = 0.0               # Heading fusionado en grados
    effective_weights: dict = field(default_factory=dict)  # Pesos efectivos usados
    global_quality: float = 0.0              # Calidad global ponderada
    num_valid_cameras: int = 0               # Cámaras válidas en esta fusión
    valid: bool = False


@dataclass
class NavCommand:
    """Comando de navegación para el AGV."""
    linear_x: float = 0.0          # Velocidad lineal (normalizada)
    angular_z: float = 0.0         # Velocidad angular (rad/s normalizado)
    state: SystemState = SystemState.STOPPED
    timestamp: float = field(default_factory=time.time)
    global_quality: float = 0.0    # Para logging/debug

    def __str__(self) -> str:
        return (
            f"NavCmd[{self.state.value}] "
            f"lin_x={self.linear_x:+.3f} "
            f"ang_z={self.angular_z:+.3f} "
            f"q={self.global_quality:.2f}"
        )

    def to_dict(self) -> dict:
        """Para futura serialización a ROS2 Twist o JSON."""
        return {
            "linear": {"x": self.linear_x, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": self.angular_z},
            "state": self.state.value,
            "global_quality": self.global_quality,
            "timestamp": self.timestamp,
        }
