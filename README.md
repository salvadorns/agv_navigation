# AGV Visual Navigation Pipeline — V1

Pipeline de navegación visual para un AGV diferencial basado en segmentación multicámara con YOLOv8.

## Descripción

Este módulo implementa una pipeline completa que:
1. Lee 3 cámaras USB (izquierda, frontal, derecha)
2. Ejecuta segmentación semántica con un modelo YOLOv8 entrenado
3. Limpia las máscaras de segmentación
4. Extrae la trayectoria central del camino por cámara
5. Evalúa la calidad de cada cámara
6. Fusiona la información con pesos adaptativos (prioridad frontal)
7. Genera comandos de navegación (`linear.x`, `angular.z`)
8. Visualiza todo el pipeline en una ventana de debug

## Requisitos

### Python
- Python 3.10+

### Dependencias
```bash
pip install ultralytics opencv-python numpy
```

Si se usa GPU (recomendado):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python numpy
```

### Hardware
- 3 cámaras USB Logitech (o compatible)
- GPU con CUDA (recomendado, no obligatorio — funciona con CPU)

## Cómo ejecutar

### Modo cámaras en vivo (default)
```bash
python main.py
```

### Detectar cámaras disponibles
```bash
python main.py --detect-cameras
```

### Especificar índices de cámaras
```bash
python main.py --cameras 0 2 4
```
(El orden es: izquierda, frontal, derecha)

### Modo video (para testing sin cámaras)
```bash
python main.py --video left.mp4 front.mp4 right.mp4
```

### Modo sin visualización
```bash
python main.py --no-vis
```

### Modelo alternativo
```bash
python main.py --model otro_modelo.pt
```

## Controles
- **q** o **ESC**: Salir
- **Ctrl+C**: Salir

## Estructura de archivos

| Archivo | Descripción |
|---------|-------------|
| `config.py` | Todos los parámetros ajustables del sistema |
| `nav_types.py` | Tipos de datos (dataclasses) para intercambio entre módulos |
| `camera_manager.py` | Lectura y gestión de las 3 cámaras USB |
| `inference.py` | Inferencia YOLOv8 y selección de máscara principal |
| `mask_processor.py` | Limpieza morfológica de máscaras |
| `trajectory.py` | Extracción de centerline y ahead point |
| `quality.py` | Score de calidad compuesto por cámara |
| `fusion.py` | Fusión multicámara con pesos adaptativos |
| `controller.py` | Control de `linear.x` y `angular.z` |
| `visualizer.py` | Ventana de debug con overlays y métricas |
| `main.py` | Entry point y loop principal |

## Parámetros ajustables

Todos los parámetros están en `config.py`. Los más importantes:

### Pesos de fusión
```python
FUSION_WEIGHT_FRONT = 0.6    # Peso base cámara frontal
FUSION_WEIGHT_LEFT  = 0.3    # Peso base cámara izquierda
FUSION_WEIGHT_RIGHT = 0.1    # Peso base cámara derecha
```

### Control
```python
CONTROL_KP_LATERAL  = 0.8    # Ganancia error lateral
CONTROL_KP_ANGULAR  = 0.4    # Ganancia error angular
CONTROL_MAX_LINEAR_X = 1.0   # Velocidad máxima (normalizada)
CONTROL_SMOOTHING_ALPHA = 0.3  # Suavizado temporal (0.1=lento, 1.0=inmediato)
```

### Seguridad
```python
SAFETY_LOW_CONFIDENCE_THRESHOLD = 0.3   # Debajo: velocidad reducida
SAFETY_STOP_THRESHOLD = 0.15            # Debajo: parada total
```

### Cámaras
```python
CAMERA_INDICES = {"left": 0, "front": 1, "right": 2}
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

## Supuestos

- El modelo `best.pt` segmenta el área navegable (camino)
- Las cámaras están montadas: izquierda (~45°), frontal (adelante), derecha (~45°)
- La clase de interés es la de mayor área segmentada (si hay múltiples clases, configurar `TARGET_CLASS_ID` en `config.py`)
- Las velocidades de salida están normalizadas a `[0, 1]` — escalar según el AGV real
- Los índices de cámaras USB son `0, 1, 2` por defecto

## Pipeline técnico

```
Frame L  ──┐
Frame F  ──┼──► YOLO Inference ──► Mask Cleanup ──► Trajectory Extraction ──► Quality Score ──┐
Frame R  ──┘                                                                                    │
                                                                                                 ▼
                                                                                         Multi-Camera Fusion
                                                                                         (adaptive weights)
                                                                                                 │
                                                                                                 ▼
                                                                                       Navigation Controller
                                                                                      (linear.x, angular.z)
                                                                                                 │
                                                                                                 ▼
                                                                                           Debug Display
                                                                                         + Console Output
```

### Fusión multicámara

Filosofía:
- **Frontal** = fuente principal de información
- **Izquierda** = secundaria fuerte, complementa la frontal
- **Derecha** = soporte suave, normalmente no decide, pero puede corregir

Los pesos se adaptan dinámicamente:
- Si la frontal pierde calidad → redistribuir a las laterales
- Si la derecha detecta una curva derecha sostenida → aumentar su peso
- Si una cámara falla → excluirla y renormalizar pesos

### Control

- `angular.z` = f(error lateral, error angular)
- `linear.x` = velocidad base × (1 − reducción × |giro|)
- Suavizado temporal (EMA) para evitar jitter
- Fallbacks: parada suave si la calidad global baja

## Integración futura con ROS2

El pipeline está diseñado para integrarse con ROS2:

```python
# Ejemplo de integración (futuro)
from main import AGVNavigationPipeline
from nav_types import NavCommand

pipeline = AGVNavigationPipeline()

# En el callback de ROS2:
result = pipeline.process_single_frame(frames)
nav_cmd = result["nav_cmd"]

# Publicar a /cmd_vel
twist = Twist()
twist.linear.x = nav_cmd.linear_x * MAX_REAL_SPEED
twist.angular.z = nav_cmd.angular_z * MAX_REAL_ANGULAR
publisher.publish(twist)
```

---

# Sistema de Telemetría ESP32 → Raspberry Pi

Módulo paralelo al pipeline de navegación. El cuatriciclo genera su propia red Wi-Fi, el ESP32 envía datos de sensores, y la Raspberry los recibe, captura una foto y embebe la telemetría en el nombre del archivo.

## Arquitectura

```
ESP32 (sensores)
    │  HTTP POST  data=48_1500_1
    ▼
Raspberry Pi 4B (Hotspot 10.42.0.1)
    │
    ├── Flask /update  ──► OpenCV cap.read()
    │
    └── /home/admin/camara_imagenes/
            2026-05-15_10-30-05_stemIn-48_stemOut-1500_ozh-1.jpg
```

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `recoleccion_de_data/servidor_telemetria.py` | Servidor Flask que recibe el POST y guarda la foto |
| `recoleccion_de_data/setup_hotspot.sh` | Script de configuración única del Hotspot + servicio systemd |

Ambos archivos corren en la Raspberry Pi y no forman parte del pipeline de navegación.

## Formato del nombre de archivo

```
YYYY-MM-DD_HH-MM-SS_stemIn-{val1}_stemOut-{val2}_ozh-{val3}.jpg
```

Ejemplo con `data=48_1500_1`:
```
2026-05-15_10-30-05_stemIn-48_stemOut-1500_ozh-1.jpg
```

## Mapeo de telemetría

| Posición en `data` | Columna lógica | Ejemplo |
|--------------------|----------------|---------|
| Valor 1 | `stemIn` | voltaje batería |
| Valor 2 | `stemOut` | RPM motores |
| Valor 3 | `ozh` | estado (0/1) |

## Setup en la Raspberry Pi

```bash
# Copiar el servidor a la ruta que espera el servicio systemd:
cp recoleccion_de_data/servidor_telemetria.py /home/admin/servidor_telemetria.py

# Una sola vez, como root:
sudo bash recoleccion_de_data/setup_hotspot.sh
```

El servicio queda activo automáticamente en cada arranque (`systemctl status telemetria`).

## Dependencias Python

```bash
pip install flask opencv-python
```

## Prueba rápida

```bash
# Desde la Raspberry — simula un POST del ESP32:
curl -X POST -d "data=48_1500_1" http://localhost:5000/update

# Verificar que el servidor responde:
curl http://localhost:5000/healthz
```

## Configuración del ESP32 (para el ingeniero electrónico)

- **SSID:** `Cuadri_Net` | **Password:** `cuadri1234`
- **Endpoint:** `http://10.42.0.1:5000/update`
- **Método:** HTTP POST
- **Body:** `data=VALOR1_VALOR2_VALOR3`

---

## Solución de problemas

| Problema | Solución |
|----------|----------|
| No detecta cámaras | Ejecutar `python main.py --detect-cameras` y ajustar `CAMERA_INDICES` en `config.py` |
| FPS muy bajo | Reducir `INFERENCE_IMGSZ` en `config.py` o usar GPU |
| Máscara ruidosa | Aumentar `MASK_MORPH_KERNEL_SIZE` o `MASK_MIN_BLOB_AREA` |
| Comando oscila mucho | Bajar `CONTROL_SMOOTHING_ALPHA` (e.g., 0.15) |
| Se detiene sin razón | Revisar `SAFETY_STOP_THRESHOLD`, puede ser muy alto |
| Cámara se desconecta | El sistema la excluye automáticamente y redistribuye pesos |
