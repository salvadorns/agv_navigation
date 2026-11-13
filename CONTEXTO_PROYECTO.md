# Contexto del proyecto: Navegación visual multicámara para AGV (tesis)

> Documento de contexto para un agente de IA. Describe el proyecto completo: objetivo, arquitectura, algoritmos, parámetros, cómo ejecutarlo, limitaciones conocidas y el contexto de la presentación de tesis. Todo lo que dice este documento se verificó contra el código del repositorio, salvo donde se aclara lo contrario.

---

## 1. Resumen

- **Qué es:** un pipeline en Python que hace que un vehículo autónomo (AGV de tracción diferencial, en la práctica un cuatriciclo o buggy) siga un camino usando solo visión por computadora.
- **Cómo funciona, en una línea:** 3 cámaras → segmentación del camino con YOLOv8 → limpieza de la máscara con OpenCV → línea central (*centerline*) y punto objetivo → puntaje de calidad por cámara → fusión ponderada de las cámaras → controlador proporcional → comandos `linear.x` y `angular.z`.
- **Estado:** prototipo de tesis (versión "V1"). Los comandos de salida están **normalizados y simulados**: todavía no se envían a motores reales. Está diseñado para integrarse más adelante con ROS2 (`/cmd_vel`).
- **Módulo aparte:** un servidor de telemetría con Flask que corre en una Raspberry Pi. Recibe datos de sensores de un ESP32 y guarda fotos con esos datos en el nombre del archivo. No está conectado al pipeline de navegación.
- **Proyecto hermano (fuera de este repo):** `Desktop\autonomouse vehicle\` contiene un **segundo pipeline**, de una sola cámara, con **behavioral cloning (PilotNet)** y un **controlador híbrido real PID + Geométrico + Pure Pursuit (+ Stanley)**. Ver la sección 12. Las afirmaciones de la presentación sobre BC y PID + Pure Pursuit corresponden a **ese** proyecto, no a este repo.
- **Idioma:** el código, los comentarios y la documentación están en **español**. El usuario es estudiante de tesis, habla español rioplatense (usa "vos") y prefiere explicaciones simples, con analogías y listas para decir en voz alta en su presentación.

---

## 2. Estructura del repositorio

```
tesis/
├── main.py                  # Punto de entrada, loop principal y CLI
├── config.py                # TODOS los parámetros ajustables
├── nav_types.py             # Dataclasses y enums compartidos entre módulos
├── camera_manager.py        # Lectura de las 3 cámaras USB
├── inference.py             # Segmentación del camino con YOLOv8 (best.pt)
├── mask_processor.py        # Limpieza morfológica de la máscara
├── trajectory.py            # Centerline, ahead point y heading
├── quality.py               # Puntaje de calidad por cámara
├── fusion.py                # Fusión multicámara con pesos adaptativos
├── controller.py            # Controlador → linear.x / angular.z
├── visualizer.py            # Ventana de debug con OpenCV
├── detector.py              # Detección de objetos (yolov8n) + modelo auxiliar 2cams
├── test_pipeline.py         # Tests sintéticos (sin cámaras)
├── recoleccion_de_data/     # Corre en la Raspberry Pi, separado del pipeline
│   ├── servidor_telemetria.py   # Flask: telemetría ESP32 → fotos
│   └── setup_hotspot.sh         # Configura el hotspot Wi-Fi y el servicio systemd
├── README.md                # Documentación del usuario
├── best.pt                  # Modelo YOLOv8 de segmentación del camino (entrenado a medida)
├── 2cams.pt                 # Modelo YOLO auxiliar (segmentación o detección)
├── video_frontal_webcam.mp4 # Video de prueba de la cámara frontal
└── .gitignore               # Ignora __pycache__, *.pyc, venv
```

**Archivo externo al repo:** `detector.py` carga `yolov8n.pt` desde una **ruta absoluta fuera del repositorio**:
`c:\Users\Novillo\OneDrive\Desktop\autonomouse vehicle\yolov8n.pt`. En otra máquina hay que cambiar esa ruta o el programa falla al iniciar.

Git: rama `main`, un solo commit ("Initial commit"), con remoto `origin`.

---

## 3. Entorno y ejecución

### Entorno verificado (máquina del usuario, Windows 11)
- **`python` NO está en el PATH**: apunta al acceso directo de la Microsoft Store. **Hay que usar el launcher `py`**.
- Intérpretes disponibles: Python 3.14 (el predeterminado de `py`) y Python 3.11.
- Paquetes instalados en 3.14: `ultralytics 8.4.30`, `opencv-python 4.13.0`, `torch 2.11.0+cpu` (**solo CPU, sin CUDA**) y `numpy`.
- La terminal es PowerShell 5.1.

### Comandos
```powershell
py main.py --front-video video_frontal_webcam.mp4            # Probado: solo cámara frontal desde video
py main.py --front-video video_frontal_webcam.mp4 --delay 100 # Cámara lenta (ms entre frames)
py main.py                                                    # 3 cámaras en vivo (índices de config)
py main.py --detect-cameras                                   # Lista los índices de cámara disponibles
py main.py --cameras 0 1 2                                    # Índices: izquierda, frontal, derecha
py main.py --video left.mp4 front.mp4 right.mp4               # 3 videos
py main.py --no-vis                                           # Sin ventana
py main.py --model otro.pt                                    # Otro modelo de segmentación
py test_pipeline.py [--test all|modules|pipeline|failover] [--scenario straight|curve_left|curve_right|narrow|lost_front|all_lost] [--frames N] [--no-vis]
```
- Para salir: `q` o `ESC` con la ventana de OpenCV seleccionada, o `Ctrl+C`.
- La consola imprime una línea cada 10 frames: `[F00010] NavCmd[NORMAL] lin_x=+0.772 ang_z=-0.165 q=0.78`.
- Si se lanza en segundo plano o con la salida redirigida, Python guarda la salida en un buffer y no la muestra hasta el final. Se soluciona con `py -u`.

### Resultado de la ejecución verificada
Con `--front-video video_frontal_webcam.mp4` procesó unos 710 frames y terminó con código de salida 0. Estuvo casi siempre en `NORMAL`, con `linear.x` alrededor de 0.7–0.8 y `q` cerca de 0.8. `angular.z` fue mayormente negativo (giro a la derecha). Hubo frames sueltos en `NO_CAMERAS` con `q=0.00`: como solo existe la cámara frontal, cuando ese frame no es "usable" el sistema se queda sin cámaras válidas y frena suavemente. La causa exacta de esos frames no se investigó.

---

## 4. Flujo de datos

```
main.py: AGVNavigationPipeline.run()  (un ciclo por frame)
│
├─ 1. CameraManager.read_all()          → Dict[CameraPosition, CameraFrame]
│
├─ para cada cámara (LEFT, FRONT, RIGHT):
│   ├─ 2. SegmentationInference.run()   → MaskResult (máscara cruda 0/255 + confianza)
│   ├─ 3. MaskProcessor.process()       → clean_mask (0/255)
│   │      (si la máscara limpia queda vacía → MaskResult.valid = False)
│   ├─ 4. TrajectoryExtractor.extract() → TrajectoryResult (centerline, ahead point, heading)
│   └─ 5. QualityEvaluator.compute()    → QualityScore (total en [0,1])
│      → CameraResult
│
├─ 6. MultiCameraFusion.fuse()          → FusionResult (ahead/heading fusionados, pesos)
├─ 7. NavigationController.compute()    → NavCommand (linear_x, angular_z, state)
├─    MultiDetector.run(frame frontal)  → DetectionOutput (cajas + overlay)  ← SOLO visual
└─ 8. DebugVisualizer.render()          → ventana "AGV Navigation Debug"
```

Si una cámara no es válida, se crea un `CameraResult` vacío y no se procesa. `process_single_frame(frames)` hace un ciclo sin visualización y devuelve `{"camera_results", "fusion_result", "nav_cmd"}`. Está pensado para integrarlo con ROS2.

`_VideoCameraManager` (en `main.py`) tiene la misma interfaz que `CameraManager`, pero lee archivos de video. Cuando el video termina, vuelve a empezar desde el principio, y redimensiona los frames a 640×480.

---

## 5. Tipos de datos (`nav_types.py`)

| Tipo | Campos principales | Notas |
|---|---|---|
| `CameraPosition` (Enum) | `LEFT="left"`, `FRONT="front"`, `RIGHT="right"` | |
| `SystemState` (Enum) | `NORMAL`, `LOW_CONFIDENCE`, `STOPPED`, `NO_CAMERAS` | |
| `CameraFrame` | `frame`, `camera_id`, `timestamp`, `valid` | `valid=False` si `frame` es None |
| `MaskResult` | `mask` (uint8 0/255), `confidence`, `valid`, `class_id` | |
| `CenterlinePoint` | `x`, `y`, `width` (en píxeles) | |
| `TrajectoryResult` | `centerline` (de abajo hacia arriba), `ahead_point` (x,y en px), `ahead_point_norm` [-1,1], `heading_angle` (°), `valid`, `num_valid_slices` | |
| `QualityScore` | `total` y 5 subscores | |
| `CameraResult` | `camera_id`, `frame`, `mask`, `clean_mask`, `trajectory`, `quality` | `is_usable = mask.valid and trajectory.valid and quality.total > 0.05` |
| `FusionResult` | `ahead_point_norm`, `heading_angle`, `effective_weights` (dict str→float), `global_quality`, `num_valid_cameras`, `valid` | |
| `NavCommand` | `linear_x`, `angular_z`, `state`, `timestamp`, `global_quality` | `to_dict()` tiene la forma de un Twist de ROS2 |

---

## 6. Módulos en detalle

### 6.1 `camera_manager.py`
- Abre las cámaras con `cv2.VideoCapture(idx, cv2.CAP_DSHOW)` (DirectShow, específico de Windows), a 640×480, 30 FPS y con `BUFFERSIZE=1` para reducir la latencia.
- Si una cámara falla, devuelve un `CameraFrame(valid=False)` y el programa sigue funcionando. Si pasan más de `CAMERA_TIMEOUT_SEC` (2 s) sin un frame válido, imprime una advertencia.
- `detect_available_cameras(max_index=10)` prueba los índices del 0 al 9.

### 6.2 `inference.py` — segmentación (`best.pt`)
- Carga el modelo YOLO una sola vez. El dispositivo se detecta solo: CUDA si está disponible, si no CPU.
- `model.predict(imgsz=640, conf=0.20, retina_masks=True)`.
- Usa `result.masks.xy`, los **polígonos ya escalados a la imagen original**. No usa `masks.data`, porque incluye el padding del *letterboxing* y deforma la máscara.
- **Combina TODAS las máscaras detectadas** en una sola, dibujando los polígonos con `cv2.fillPoly`. Si `TARGET_CLASS_ID` tiene un valor, filtra por esa clase. Por defecto es `None`.
  - ⚠️ El docstring dice que elige la máscara que maximiza área × confianza, pero el código las fusiona todas. Lo correcto es lo que hace el código.
- La confianza es el promedio de las confianzas, ponderado por el área de cada *bounding box*.
- Si la máscara combinada tiene menos de `MASK_MIN_BLOB_AREA` (800 px), devuelve `valid=False`.

### 6.3 `mask_processor.py` — limpieza con OpenCV
Los pasos, en orden:
1. **Recorte superior:** pone en 0 el 25% superior de la máscara (`MASK_TOP_CROP_RATIO`), donde suelen estar el cielo y los edificios. ⚠️ Modifica la máscara de entrada directamente.
2. **Binarización:** `threshold(127)`.
3. **Cierre morfológico** (dilatar y después erosionar) con un kernel elíptico de 15 px: tapa huecos chicos.
4. **Segundo cierre** con un kernel elíptico de 35 px: tapa huecos más grandes.
5. **Relleno de contornos:** `findContours(RETR_EXTERNAL)` y `drawContours(FILLED)` rellenan los agujeros internos.
6. **Extrusión hacia abajo** (`_extrude_bottom`): si la máscara no llega al borde inferior (queda a más de 10 px) y su fila más baja está en la mitad inferior de la imagen, dibuja un rectángulo desde esa fila hasta el borde. Compensa que el dataset se etiquetó con polígonos que no llegaban al borde.
7. **Componente principal:** `connectedComponentsWithStats` (conectividad 8). Conserva solo la región blanca más grande, y si mide menos de 800 px devuelve una máscara vacía.

No hay un desenfoque ni un suavizado de la imagen. El "suavizado" del sistema está en el controlador (EMA).

### 6.4 `trajectory.py` — centerline, ahead point y heading
- **ROI:** ignora el 15% superior de la imagen (`TRAJECTORY_TOP_IGNORE_RATIO=0.15`).
- **10 franjas horizontales** (`TRAJECTORY_NUM_SLICES`), recorridas de abajo hacia arriba. En cada franja:
  - `col_presence = np.any(stripe > 127, axis=0)`: una columna "tiene camino" si cualquier fila de la franja es blanca.
  - `left_edge` y `right_edge` son la **primera y la última columna con blanco**.
  - `center_x = (left_edge + right_edge) / 2`.
  - La franja se descarta si tiene menos de 10 columnas activas o un ancho menor a 10 px.
- Si quedan menos de 2 puntos, el resultado es `valid=False`.
- **Ahead point (el punto rojo):** se ubica en el índice `0.4 × (n−1)` de la centerline (`TRAJECTORY_LOOKAHEAD_RATIO=0.4`), con interpolación lineal entre los dos puntos vecinos. Con 10 puntos cae en el índice 3.6, entre el 4.º y el 5.º desde abajo.
  - `ahead_point_norm = (x − W/2)/(W/2)`, limitado a [-1, 1]. −1 = izquierda, 0 = centro, +1 = derecha. En la pantalla se muestra como **"A"**.
- **Heading (H):** regresión lineal `x = m·y + b` sobre la **mitad superior** de la centerline (al menos 3 puntos). `angle = degrees(atan(−m))`, limitado a ±60°. Positivo = el camino va hacia la derecha. Si hay menos de 3 puntos, vale 0.

### 6.5 `quality.py` — puntaje de calidad (Q)
Si la máscara o la trayectoria no son válidas, Q = 0 y se agrega `None` al historial. Si no:

| Factor | Peso | Cálculo |
|---|---|---|
| Área | 0.30 | proporción de píxeles de camino en el ROI ÷ 0.6, limitada a 1 |
| Continuidad | 0.25 | `num_valid_slices / 10` |
| Confianza | 0.20 | `min(1, mask.confidence)` |
| Consistencia | 0.15 | `max(0, 1 − std(x de la centerline) / ancho_promedio)`; vale 0.5 si hay menos de 3 puntos |
| Estabilidad | 0.10 | `max(0, 1 − 2 × media(|Δahead_norm|))` sobre los últimos 5 frames; vale 0.8 si hay menos de 2 |

El total se limita a [0, 1].

### 6.6 `fusion.py` — fusión multicámara
1. Solo se usan las cámaras con `is_usable`. Si no hay ninguna, el resultado es `valid=False`.
2. **Pesos base:** frontal 0.6, izquierda 0.3, derecha 0.1. Las cámaras no usables valen 0.
3. Si la frontal tiene Q < 0.3, se le resta `w_front × (1 − Q/0.3)` y eso se reparte en partes iguales entre las laterales usables.
4. **Curva a la derecha:** si `ahead_norm` de la cámara derecha es menor a −0.15 durante 8 frames seguidos, el peso de la derecha pasa a 0.25 y se le resta esa diferencia a la frontal (que nunca baja de 0.1).
5. **Modulación por calidad:** `w *= (0.5 + 0.5·Q)`.
6. Se normalizan los pesos para que sumen 1.
7. `fused_ahead = Σ w·ahead_norm`, `fused_heading = Σ w·heading` y `global_quality = Σ w·Q`.

Con una sola cámara usable, su peso es 1.0 y el resultado fusionado es igual al de esa cámara.

### 6.7 `controller.py` — controlador
**Es un controlador proporcional (P). No tiene términos I ni D, y no es un pure pursuit geométrico.** Comparte con pure pursuit solo la idea de perseguir un punto adelantado.
1. **Seguridad:** si la fusión no es válida o no hay cámaras, estado `NO_CAMERAS` y frenado. Si `global_quality < 0.15`, estado `STOPPED` y frenado.
2. **Errores:** `e_lat = fused_ahead`; `e_ang = clamp(fused_heading / 45, −1, 1)`.
3. **Giro:** `angular_z = −(0.8·e_lat + 0.4·e_ang)`, limitado a ±1. **Negativo = girar a la derecha** (convención de ROS).
4. **Velocidad:** `linear_x = 1.0 × (1 − 0.6·|angular_z|)`, con un mínimo de 0.1.
5. Si `global_quality < 0.3`, estado `LOW_CONFIDENCE` y `linear_x × 0.5`.
6. **Suavizado EMA:** `s = 0.3·raw + 0.7·s_anterior`, tanto para la velocidad como para el giro.
7. **Frenado suave** (`_stop`): multiplica ambos valores por 0.3 en cada frame y los fija en 0 cuando bajan de 0.01.

Ejemplo con datos reales de una captura (A=+0.07, H=43.2°): `angular_z = −(0.056 + 0.384) = −0.44`, o sea giro a la derecha. Velocidad sin suavizar: `1 − 0.6×0.44 = 0.74`.

### 6.8 `detector.py` — detección de objetos (solo visual)
- `yolov8n.pt` (COCO). Muestra solo estas clases: 0 Persona (rojo), 2 Auto (azul), 3 Moto, 5 Bus y 7 Camión, con el porcentaje de confianza.
- `2cams.pt`: si devuelve máscaras, se pintan como un overlay **naranja**, solo fuera del camino. Si devuelve cajas, se dibujan en naranja.
- Se ejecuta **solo sobre la cámara frontal** y **solo se usa para dibujar**. ⚠️ **No llega al controlador: el vehículo no frena ante personas ni autos.**

### 6.9 `visualizer.py` — ventana de debug
El layout es una fila de 3 paneles (LEFT, FRONT, RIGHT) de 640×480 más un panel inferior de información de 100 px. Todo se escala por `VIS_SCALE=0.8`.

**Elementos de cada panel de cámara:**
| Elemento | Significado |
|---|---|
| Zona **verde** semitransparente | máscara limpia del camino |
| Overlay **naranja** | segmentación de `2cams.pt` fuera del camino (solo en la frontal) |
| Cajas con etiqueta "Auto 37%", "Persona 44%" | detecciones de `yolov8n` |
| **Línea amarilla** con puntos | centerline (punto medio de cada franja) |
| **Punto rojo** (relleno y con anillo) | ahead point de esta cámara |
| **Círculo celeste "FUSED"** (solo en la frontal) | ahead point fusionado. La x es `(A_fus+1)/2·W`; la **y está fija en la mitad de la imagen y no significa nada** |
| `H=…deg` (arriba a la derecha) | heading de esta cámara |
| `A=…` (arriba a la derecha) | ahead_point_norm de esta cámara |
| `Q=…` (abajo a la izquierda) | calidad: verde > 0.7, amarillo > 0.4, rojo ≤ 0.4 |
| `W=…` (abajo a la izquierda) | peso efectivo en la fusión |
| Borde rojo + "NO DATA" | la cámara existe pero no es usable |
| Panel negro + borde rojo + "DISCONNECTED" | la cámara no existe (por ejemplo las laterales en modo `--front-video`) |

**Panel inferior:** `FUSION: ahead heading quality cameras=N/3`, `CMD: linear.x angular.z`, `STATE` (coloreado según el estado), `FPS` y `Weights`, y una barra de dirección (el punto se mueve hacia el lado del giro: verde si el estado es NORMAL, rojo si no).

`render()` devuelve el canvas completo, aunque hoy no se graba. Para grabarlo se podría agregar un `cv2.VideoWriter`.

### 6.10 `test_pipeline.py`
- `modules`: prueba cada módulo con una máscara trapezoidal sintética, sin usar YOLO.
- `failover`: va matando cámaras en este orden: todas → sin la derecha → solo la frontal → ninguna. Al final verifica que `linear_x == 0` después del frenado.
- `pipeline`: frames sintéticos (un trapecio de camino con cielo) pasados por YOLO. Si YOLO no detecta nada, usa como respaldo un umbral sobre la imagen en grises. Escenarios: straight, curve_left, curve_right, narrow, lost_front y all_lost.

---

## 7. Parámetros clave (`config.py`)

| Grupo | Parámetro | Valor |
|---|---|---|
| Cámaras | `CAMERA_INDICES` | left 0, front 1, right 2 |
| | `CAMERA_WIDTH/HEIGHT/FPS` | 640 / 480 / 30 |
| | `CAMERA_TIMEOUT_SEC` | 2.0 |
| Inferencia | `MODEL_PATH` | `best.pt`, junto a config.py |
| | `INFERENCE_IMGSZ`, `INFERENCE_CONF_THRESHOLD` | 640, 0.20 |
| | `INFERENCE_DEVICE`, `TARGET_CLASS_ID` | None (autodetección), None (todas) |
| Máscara | `MASK_BINARY_THRESHOLD` | 0.5 (definido, pero el código usa 127 directamente) |
| | `MASK_MORPH_KERNEL_SIZE`, `MASK_FILL_KERNEL_SIZE` | 15, 35 |
| | `MASK_MIN_BLOB_AREA`, `MASK_TOP_CROP_RATIO` | 800, 0.25 |
| Trayectoria | `TRAJECTORY_NUM_SLICES`, `TOP_IGNORE_RATIO` | 10, 0.15 |
| | `TRAJECTORY_LOOKAHEAD_RATIO`, `MIN_PATH_WIDTH` | 0.4, 10 |
| Calidad | pesos área/continuidad/confianza/consistencia/estabilidad | 0.30/0.25/0.20/0.15/0.10 |
| | `QUALITY_MAX_AREA_RATIO`, `TEMPORAL_WINDOW` | 0.6, 5 |
| Fusión | pesos frontal/izquierda/derecha | 0.6/0.3/0.1 |
| | `FUSION_FRONT_QUALITY_THRESHOLD` | 0.3 |
| | `FUSION_EXCLUDE_QUALITY_THRESHOLD` | 0.05 (**no se usa**: `is_usable` tiene 0.05 escrito directamente en el código) |
| | curva derecha: umbral / persistencia / peso | −0.15 / 8 frames / 0.25 |
| Control | `KP_LATERAL`, `KP_ANGULAR` | 0.8, 0.4 |
| | `MAX_LINEAR_X`, `MIN_LINEAR_X` | 1.0, 0.1 |
| | `TURN_REDUCTION_FACTOR`, `SMOOTHING_ALPHA`, `MAX_ANGULAR_Z` | 0.6, 0.3, 1.0 |
| Seguridad | `SAFETY_LOW_CONFIDENCE_THRESHOLD`, `STOP_THRESHOLD` | 0.3, 0.15 |
| | `SAFETY_MIN_VALID_CAMERAS` | 1 |
| Visualización | `VIS_SCALE`, `VIS_ENABLED` | 0.8, True |

---

## 8. Módulo de telemetría (Raspberry Pi), independiente del pipeline

- **Hardware:** una Raspberry Pi 4B crea una red Wi-Fi propia. Un ESP32 con sensores se conecta a esa red y le envía datos por HTTP.
- Los dos archivos están en la carpeta `recoleccion_de_data/`.
- `setup_hotspot.sh` (se ejecuta una vez con `sudo`): instala Flask, OpenCV y NetworkManager; crea el hotspot `Cuadri_Net` con IP fija `10.42.0.1/24`; y registra el servicio systemd `telemetria`, que ejecuta `/home/admin/servidor_telemetria.py`.
- `servidor_telemetria.py` (Flask, `0.0.0.0:5000`):
  - `POST /update` con el cuerpo `data=V1_V2_V3` (por ejemplo `48_1500_1`): captura un frame de la cámara (prueba el índice 0 y después el 1) y lo guarda en `/home/admin/camara_imagenes/YYYY-MM-DD_HH-MM-SS_stemIn-V1_stemOut-V2_ozh-V3.jpg`.
  - `GET /healthz` responde `{"status": "running"}`.
  - Según el README, el mapeo es: `stemIn` = voltaje de batería, `stemOut` = RPM de los motores, `ozh` = estado (0/1).
- La contraseña del hotspot está en texto plano en el README y en el script.

---

## 9. Limitaciones, errores conocidos y diferencias con la documentación

1. **Centerline sesgada cerca del horizonte.** Como el centro se calcula entre el primer y el último píxel verde de toda la franja, cualquier zona extra segmentada como camino (por ejemplo una playa de estacionamiento o una intersección al fondo) corre el centro. En las capturas reales, el tramo superior de la línea amarilla se desvía mucho a la derecha y eso **infla H** (se vieron valores de 43–46° en un camino casi recto). El punto rojo, que está al 40% desde abajo, casi no se ve afectado. Posibles soluciones:
   - tomar solo el tramo continuo de verde más cercano al centro de la franja anterior (la más robusta);
   - usar el promedio de las columnas activas;
   - subir `TRAJECTORY_TOP_IGNORE_RATIO`.
2. **Las detecciones de objetos no afectan el control** (ver 6.8). No hay frenado ante obstáculos.
3. **El controlador de este repo es solo P.** No hay PID completo ni pure pursuit geométrico (no se calcula la curvatura del arco). El PID completo y el Pure Pursuit están en el proyecto hermano (sección 12).
4. **No hay *behavioral cloning* en este repositorio.** Está en el proyecto hermano (sección 12), y los dos pipelines no están conectados.
5. Hay **una ruta absoluta hardcodeada** a `yolov8n.pt` en `detector.py`.
6. El docstring de `inference.py` no coincide con el código (ver 6.2).
7. Hay parámetros de config definidos pero sin usar: `FUSION_EXCLUDE_QUALITY_THRESHOLD` y `MASK_BINARY_THRESHOLD`.
8. `MASK_TOP_CROP_RATIO` (0.25) es mayor que `TRAJECTORY_TOP_IGNORE_RATIO` (0.15), así que la franja superior de la trayectoria cae en una zona ya borrada y siempre está vacía.
9. Hay una pequeña diferencia entre `run()` y `process_single_frame()` cuando la máscara limpia queda vacía. `process_single_frame()` crea `MaskResult(valid=False)` sin conservar ni la máscara ni la confianza.
10. **Rendimiento:** torch funciona solo en CPU y en cada frame se ejecutan 3 modelos YOLO (best + yolov8n + 2cams), así que los FPS son bajos.
11. Las velocidades de salida están normalizadas: falta escalarlas a valores reales del vehículo (`MAX_REAL_SPEED`).
12. No hay tests automatizados con pytest: `test_pipeline.py` es un script con `assert`.

---

## 10. Contexto de la presentación de tesis

El usuario está preparando una presentación oral. Hay una diapositiva titulada **"04 · Funciones y resultados técnicos II — Percepción, trayectoria y control en un único pipeline"** con este contenido:
- Encabezado: `CÁMARAS → MÁSCARA → CENTERLINE → CONTROL → COMANDOS`.
- Viñetas: "OpenCV + filtros morfológicos + suavizado → centerline", "Behavioral cloning para bulevar y calle adoquinada", "Control híbrido PID + Pure Pursuit → salidas simuladas de motor".
- Una captura del panel FRONT (camino verde, línea amarilla, punto rojo, círculo FUSED, detección "Persona 44%", H=43.2deg, A=+0.07) y un video en un celular que muestra el prototipo (un buggy chico) recorriendo un bulevar.

Hay otra diapositiva con el diagrama de los 8 módulos (camera_manager → inference → mask_processor → trajectory / quality → **fusion** → **controller** → visualizer), con `fusion.py` y `controller.py` resaltados.

**⚠️ Cómo se relacionan las diapositivas con el código (señalarlo con tacto si piden material para la presentación):**
- **"Control híbrido PID + Pure Pursuit"** es correcto **para el proyecto hermano** (`autonomouse vehicle/pipeline/control.py`). En el pipeline de 3 cámaras de este repo, el control es solo proporcional. No hay que mezclar los dos al explicarlo.
- **"Behavioral cloning"** existe (PilotNet, en el proyecto hermano). **"Para bulevar y calle adoquinada"** no está programado como un modo por tipo de calle: el BC toma el control automáticamente cuando la confianza geométrica es baja.
- La diapositiva dice "suavizado → centerline", pero en este repo no hay suavizado de imagen: el suavizado está en los **comandos** (EMA). En el proyecto hermano, el BC sí aplica un GaussianBlur 3×3 como preprocesamiento.
- La "Persona 44%" que se ve en la captura **no hace frenar** al vehículo, en ninguno de los dos pipelines. En el `main.py` del proyecto hermano solo dispara un banner de alerta visual si los pies de la persona caen sobre el camino.

**Explicaciones simples que el usuario ya aprobó** (conviene reutilizar el mismo estilo):
- *Punto rojo:* "como cuando manejamos: no miramos el capó, miramos unos metros adelante".
- *Círculo FUSED:* "una votación con pesos: la frontal tiene el voto más fuerte y una cámara que ve mal pesa menos".
- *A y H:* "A = ¿estoy centrado?; H = ¿el camino dobla más adelante?".
- *Filtros morfológicos:* "pasar un pincel grueso para tapar los agujeros y después recortar el borde para que la mancha vuelva a su tamaño".
- *Desvío de la línea amarilla:* "no es que el camino dobla: es que el programa ve una calle más ancha de lo que es".
- *Velocidad y suavizado:* "cuanto más gira, más despacio va; sin volantazos".

---

## 11. Glosario

| Término | Significado en este proyecto |
|---|---|
| AGV | *Automated Guided Vehicle*: el vehículo autónomo |
| Máscara | imagen en blanco y negro donde blanco = camino |
| Centerline | línea central del camino (amarilla) |
| Ahead point / lookahead | punto objetivo adelantado sobre la centerline (rojo) |
| A / `ahead_point_norm` | posición horizontal del objetivo en [-1, 1] |
| H / heading | ángulo hacia el que se orienta el camino, en grados |
| Q | puntaje de calidad de una cámara, en [0, 1] |
| W | peso de la cámara en la fusión |
| FUSED | objetivo combinado de todas las cámaras (círculo celeste) |
| `linear.x` | velocidad hacia adelante (normalizada a [0, 1]) |
| `angular.z` | velocidad de giro (negativo = derecha) |
| EMA | media móvil exponencial, usada para suavizar los comandos |
| Cierre morfológico | dilatar y después erosionar: rellena huecos |
| ROI | región de interés: la parte de la imagen que se analiza |
| BC | *Behavioral cloning*: una red aprende a imitar el giro de un conductor humano a partir de imágenes |
| PilotNet | CNN de NVIDIA (2016) que va de la imagen al ángulo de dirección |
| Stanley | controlador de seguimiento de trayectoria que combina heading y error lateral |

---

## 12. Proyecto hermano: `Desktop\autonomouse vehicle\` (BC + control híbrido)

Es una carpeta **fuera del repo `tesis`**, con su propio `venv` y su propio `requirements_pipeline.txt` (opencv, **tensorflow**, ultralytics y numpy). La cabecera de los scripts dice "AUTHOR: Salvador Novillo Saravia". Trabaja con **una sola cámara**, procesa un video y escribe un video anotado (`output_nav*.mp4`). No muestra una ventana en vivo.

Detalle completo del BC (dataset, entrenamiento, métricas, preguntas abiertas): ver [CHECKLIST_BEHAVIORAL_CLONING.md](CHECKLIST_BEHAVIORAL_CLONING.md).

### Archivos
| Archivo | Rol |
|---|---|
| `main.py` | Pipeline 1: segmentación + BC + detección de personas. Solo visualización (centroide del camino, flecha de giro del BC y alerta si hay una persona sobre el camino) |
| `main_nav.py` | **Pipeline 2, "geometry-first"**: segmentación → geometría → controlador híbrido → fusión con BC → filtros de seguridad |
| `pipeline/segmentation.py` | YOLO-seg (`2cams.pt` o `best.pt`), `ROAD_CLASS_ID = 0`, confianza 0.4 |
| `pipeline/geometry.py` | Bordes del camino en 8 líneas de escaneo, polinomios de grado 2, error lateral, heading, curvatura, lookahead y confianza |
| `pipeline/control.py` | Estrategias PID, Geométrica, Pure Pursuit y Stanley + fusión con BC + filtros |
| `pipeline/steering.py` | `SteeringModule`: inferencia del modelo BC con suavizado de media móvil (10 frames) |
| `pipeline/detection.py` | yolov8n, solo personas |
| `pipeline/visualization.py`, `nav_visualization.py` | overlays de los videos |
| `Untitled0.ipynb` | Entrenamiento del BC en Colab |
| `model_best_val_loss.keras` | Modelo BC entrenado |
| `labels_generadas.csv` | 5351 etiquetas extraídas de los nombres de archivo |

### Uso
```
python main_nav.py --input video.mp4 --seg 2cams.pt --steer model_best_val_loss.keras --output output_nav.mp4
  [--seg-conf 0.4] [--det-conf 0.45] [--smooth 10] [--pid-kp 0.6] [--pid-ki 0.05] [--pid-kd 0.1]
```
Sin `--steer`, funciona solo con geometría. En la máquina del usuario hay que usar `py` o el `venv` de esa carpeta. No se verificó que corra.

### Geometría (`geometry.py`)
- **8 líneas de escaneo** en el 25, 35, 45, 55, 65, 75, 82 y 88% de la altura. En cada una se buscan los bordes izquierdo y derecho del camino (ancho mínimo 20 px).
- **Respaldo de un solo borde:** si un borde toca el límite de la imagen, se estima a partir del otro usando el ancho promedio de los últimos 60 frames.
- **Polinomios** `x = f(y)` de grado 2 para el borde izquierdo, el derecho y el centro (con al menos 3 puntos).
- **Error lateral** en la fila del 80%: `(cx − W/2)/(W/2)`.
- **Heading:** `atan2(dx, dy)` entre la fila del 80% y la del 50%.
- **Curvatura:** la del polinomio central.
- **Lookahead:** en la fila del 60%.
- **Confianza** = `(0.60·franjas_válidas/8 + 0.15·izq_visible + 0.15·der_visible + 0.10·polinomio_ok − 0.15·respaldo_un_borde) × confianza_segmentación`.
- Convención: **positivo = girar a la DERECHA**. Es al revés que `angular.z` en el repo `tesis`.

### Controlador (`control.py`)
| Estrategia | Fórmula |
|---|---|
| PID | `Kp·e + Ki·Σe + Kd·Δe` sobre el error lateral (0.60 / 0.05 / 0.10), con anti-windup (integral limitada a ±1) |
| Geométrica | `0.5·e_lat + 0.4·(heading/45°) + 0.3·tanh(50·curvatura)` |
| Pure Pursuit | `atan2(dx, dy)` hacia el punto de lookahead, normalizado (±45° = ±1) |
| Stanley | `heading/45° + atan2(0.3·e_lat, 0.3)/45°`. **Se calcula pero no entra en la mezcla** |
| **Compuesto** | `0.40·Geométrica + 0.35·PID + 0.25·PurePursuit` |

**Fusión con BC según la confianza:** si es ≥ 0.70, 100% compuesto; si es ≤ 0.30, 100% BC; en el medio, mezcla lineal. Si |compuesto − BC| > 0.45, se marca `diverged` (es solo un aviso).

**Filtros de seguridad:** saturación ±1 → limitador de cambio (0.12 por frame) → EMA (α = 0.35). La salida es `steer_final` ∈ [−1, 1]. Para pasarla a grados: `× MAX_STEERING_ANGLE_DEG`.

**Diferencia con el repo `tesis`:** este controlador **no genera una velocidad lineal**, solo el giro. Tampoco tiene estados de seguridad (NORMAL, STOPPED, etc.) ni fusión multicámara.

### Behavioral cloning (resumen)
- **Modelo:** PilotNet de NVIDIA en Keras, con 252.219 parámetros. Entrada YUV de 66×200 (recorte de las filas 150 a 350, desenfoque 3×3, /255). Salida: un ángulo de dirección continuo, `steering_raw/100`.
- **Datos:** 5351 imágenes en 4 carpetas. La etiqueta sale del nombre del archivo, `ID_STEER_X_Y.jpg`. Después del balanceo (25 bins, máximo 400 por bin) quedaron 2107. División 80/20: 1685 / 422.
- **Aumentos:** pan, zoom, brillo y espejado con inversión del ángulo.
- **Entrenamiento:** MSE, Adam con lr 1e-4, lote 64, en Colab. Se cortó por early stopping en la época 17 de 30. **Mejor val_loss: 0.0755** (RMSE ≈ 0.27 normalizado).
- **Prueba sobre video:** predicciones en [−0.29, +0.15], con promedio −0.014. Son giros chicos.
- **Preguntas abiertas:** las unidades de `steering_raw`, el significado de los otros campos del nombre de archivo, qué calle hay en cada carpeta y si se probó en el vehículo.
