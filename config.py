"""
config.py — Configuración central del módulo de navegación visual AGV.

Todos los parámetros ajustables del pipeline están aquí.
Modificar estos valores para calibrar el sistema al entorno real.
"""

import os

# ==============================================================================
# CÁMARAS
# ==============================================================================

# Índices USB de las cámaras. Ajustar según el orden de conexión.
# En Windows, los índices suelen ser 0, 1, 2, etc.
CAMERA_INDICES = {
    "left": 0,
    "front": 1,
    "right": 2,
}

# Resolución de captura (ancho x alto). 640x480 ofrece buen balance FPS/calidad.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Timeout en segundos para considerar una cámara como fallida
CAMERA_TIMEOUT_SEC = 2.0


# ==============================================================================
# INFERENCIA YOLO
# ==============================================================================

# Ruta al modelo de segmentación entrenado
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

# Tamaño de imagen para inferencia (mayor = más preciso pero más lento)
INFERENCE_IMGSZ = 640

# Umbral de confianza mínimo para aceptar una detección (Bajar evita que la calle desaparezca)
INFERENCE_CONF_THRESHOLD = 0.20

# Device: "cuda", "cpu", o None para autodetección
INFERENCE_DEVICE = None

# Clase de interés para el camino. None = seleccionar la de mayor área.
# Si el modelo tiene múltiples clases, poner aquí el ID (int) de la clase "camino".
TARGET_CLASS_ID = None


# ==============================================================================
# POSTPROCESO DE MÁSCARA
# ==============================================================================

# Umbral de binarización (las máscaras YOLO son soft en [0,1])
MASK_BINARY_THRESHOLD = 0.5

# Tamaño del kernel para cierre morfológico (impar, mayor = más agresivo)
MASK_MORPH_KERNEL_SIZE = 15

# Tamaño del kernel para relleno de huecos (segundo cierre más grande)
MASK_FILL_KERNEL_SIZE = 35

# Área mínima en píxeles para conservar un blob (elimina ruido)
MASK_MIN_BLOB_AREA = 800

# Fracción superior de la imagen a ignorar en la máscara (el cielo/edificios no son calle)
# 0.0 = sin recorte: se conserva todo lo que el modelo reconoce como calle.
MASK_TOP_CROP_RATIO = 0.0


# ==============================================================================
# EXTRACCIÓN DE TRAYECTORIA
# ==============================================================================

# Número de franjas horizontales para analizar el camino
TRAJECTORY_NUM_SLICES = 10

# Ratio de la imagen a ignorar desde arriba (cielo/horizonte)
# 0.3 = ignorar el 30% superior de la imagen
TRAJECTORY_TOP_IGNORE_RATIO = 0.15

# Ratio de lookahead para el ahead point (0.0 = base, 1.0 = tope del ROI)
# 0.4 = mirar al 40% de altura dentro del ROI
TRAJECTORY_LOOKAHEAD_RATIO = 0.4

# Ancho mínimo del camino en píxeles para considerar una franja válida
TRAJECTORY_MIN_PATH_WIDTH = 10


# ==============================================================================
# QUALITY SCORE
# ==============================================================================

# Pesos de cada factor en el score compuesto
QUALITY_WEIGHT_AREA = 0.30
QUALITY_WEIGHT_CONTINUITY = 0.25
QUALITY_WEIGHT_CONFIDENCE = 0.20
QUALITY_WEIGHT_CONSISTENCY = 0.15
QUALITY_WEIGHT_STABILITY = 0.10

# Área máxima esperada de la máscara (fracción del total de píxeles en ROI)
# Usado para normalizar el factor de área
QUALITY_MAX_AREA_RATIO = 0.6

# Ventana temporal para estabilidad (número de frames)
QUALITY_TEMPORAL_WINDOW = 5


# ==============================================================================
# FUSIÓN MULTICÁMARA
# ==============================================================================

# Pesos base por cámara
FUSION_WEIGHT_FRONT = 0.6
FUSION_WEIGHT_LEFT = 0.3
FUSION_WEIGHT_RIGHT = 0.1

# Umbral de calidad debajo del cual se redistribuye peso de la cámara frontal
FUSION_FRONT_QUALITY_THRESHOLD = 0.3

# Umbral de calidad debajo del cual una cámara se excluye de la fusión
FUSION_EXCLUDE_QUALITY_THRESHOLD = 0.05

# Detección de curva derecha: desplazamiento mínimo normalizado del ahead point
# Un valor negativo del ahead point normalizado en la cámara derecha indica
# que el camino se desvía hacia la derecha del AGV
FUSION_RIGHT_CURVE_THRESHOLD = -0.15

# Frames consecutivos necesarios para confirmar curva derecha
FUSION_RIGHT_CURVE_PERSISTENCE = 8

# Peso aumentado de la cámara derecha al detectar curva derecha
FUSION_RIGHT_CURVE_WEIGHT = 0.25


# ==============================================================================
# CONTROLADOR DE NAVEGACIÓN
# ==============================================================================

# Ganancias del controlador
CONTROL_KP_LATERAL = 0.8       # Ganancia proporcional para error lateral
CONTROL_KP_ANGULAR = 0.4       # Ganancia proporcional para error angular

# Velocidades
CONTROL_MAX_LINEAR_X = 1.0     # Velocidad lineal máxima (normalizada)
CONTROL_MIN_LINEAR_X = 0.1     # Velocidad lineal mínima (para no detenerse en curvas)

# Factor de reducción de velocidad por giro
# linear.x = max_vel * (1 - reduction * |angular.z|)
CONTROL_TURN_REDUCTION_FACTOR = 0.6

# Suavizado temporal EMA (Exponential Moving Average)
# alpha = 1.0 → sin suavizado, alpha = 0.1 → muy suave
CONTROL_SMOOTHING_ALPHA = 0.3

# Límite máximo de angular.z (para evitar giros violentos)
CONTROL_MAX_ANGULAR_Z = 1.0


# ==============================================================================
# SEGURIDAD Y FALLBACKS
# ==============================================================================

# Calidad global debajo de la cual se reduce velocidad al 50%
SAFETY_LOW_CONFIDENCE_THRESHOLD = 0.3

# Calidad global debajo de la cual se detiene completamente
SAFETY_STOP_THRESHOLD = 0.15

# Número mínimo de cámaras válidas para operar
SAFETY_MIN_VALID_CAMERAS = 1


# ==============================================================================
# VISUALIZACIÓN
# ==============================================================================

# Escala de redimensión para la ventana de debug (1.0 = tamaño completo)
VIS_SCALE = 0.8

# Color del overlay de máscara (BGR)
VIS_MASK_COLOR = (0, 200, 0)       # Verde
VIS_MASK_ALPHA = 0.35              # Transparencia del overlay

# Colores para elementos de debug (BGR)
VIS_CENTERLINE_COLOR = (0, 255, 255)   # Amarillo
VIS_AHEAD_POINT_COLOR = (0, 0, 255)    # Rojo (por cámara)
VIS_FUSED_POINT_COLOR = (255, 255, 0)  # Cyan (fusionado)
VIS_TEXT_COLOR = (255, 255, 255)       # Blanco
VIS_WARNING_COLOR = (0, 100, 255)      # Naranja

# Mostrar ventana de debug
VIS_ENABLED = True
