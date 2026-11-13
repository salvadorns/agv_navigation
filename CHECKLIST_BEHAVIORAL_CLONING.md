# Información pendiente: Behavioral Cloning

Para escribir el apartado de *behavioral cloning* con el mismo nivel técnico que el resto, necesito la información de abajo.

> **Mínimo indispensable:** los puntos **2 (Modelo)**, **3 (Dataset)** y **6 (Integración)**. Con eso se puede describir el enfoque de forma honesta y dejar las métricas como pendientes.

> ✅ **Actualización:** encontré el código en `Desktop\autonomouse vehicle\`. Las respuestas de abajo salen de leer ese código y la notebook. Lo que **no** se puede saber leyendo el código quedó marcado con ❓.

---

## 1. El código

- [x] **¿Dónde está?** Puede ser otro repositorio, otra rama, una notebook de Colab o la compu de Tobías.

**Respuesta:** en `C:\Users\Novillo\OneDrive\Desktop\autonomouse vehicle\`, fuera del repositorio `tesis`.

| Archivo | Qué es |
|---|---|
| `Untitled0.ipynb` | Notebook de **Google Colab** con todo el entrenamiento (53 MB porque tiene imágenes adentro) |
| `model_best_val_loss.keras` | Modelo entrenado (3 MB): el mejor según la pérdida de validación |
| `labels_generadas.csv` | Etiquetas generadas a partir de los nombres de las imágenes (5351 filas) |
| `pipeline/steering.py` | Módulo de inferencia (`SteeringModule`) |
| `pipeline/control.py` | Controlador híbrido: PID + Geométrico + Pure Pursuit + Stanley, y fusión con BC |
| `pipeline/geometry.py` | Geometría del camino: bordes, polinomios, error lateral, heading, curvatura, confianza |
| `main.py` | Pipeline 1: segmentación + BC + detección, solo visualización |
| `main_nav.py` | **Pipeline 2: geometría primero, con BC como respaldo** (es el que integra todo) |
| `output_nav*.mp4`, `output.mp4` | Videos de salida de las corridas |

Las imágenes originales y el video de prueba (`videotest.mp4`) están en Google Drive:
`MyDrive/The-Complete-Self-Driving-Car-Course---Applied-Deep-Learning-master/9. Behavioral Cloning/`

---

## 2. El modelo ⭐ *(indispensable)*

- [x] **Arquitectura**
- [x] **Entrada**
- [x] **Salida**

**Respuesta:**
- **Arquitectura: PilotNet de NVIDIA** (Bojarski et al., 2016), una CNN entrenada desde cero, sin pesos preentrenados.

  | Capa | Configuración |
  |---|---|
  | Conv2D | 24 filtros 5×5, stride 2, ELU |
  | Conv2D | 36 filtros 5×5, stride 2, ELU |
  | Conv2D | 48 filtros 5×5, stride 2, ELU |
  | Conv2D | 64 filtros 3×3, ELU |
  | Conv2D | 64 filtros 3×3, ELU |
  | Flatten | 1152 |
  | Dense | 100 → 50 → 10 (ELU) → **1** (lineal) |

  **Total: 252.219 parámetros** (unos 985 KB).
- **Entrada:** una sola cámara (frontal). El preprocesamiento es el mismo en el entrenamiento y en la inferencia:
  1. Recorte de las filas `[150:350]`. Por eso el frame tiene que medir al menos 350 px de alto.
  2. Conversión RGB → **YUV**, como en el paper de NVIDIA.
  3. Desenfoque gaussiano 3×3.
  4. Redimensión a **200×66**.
  5. Normalización `/255`.
- **Salida:** **un valor continuo** (regresión), el ángulo de dirección normalizado, aproximadamente en [−1, 1].
  - `steering = steering_raw / 100`. Para mostrarlo se multiplica de nuevo × 100.
  - ❓ **Unidades de `steering_raw`:** el código no lo dice. El rango observado va de −109 a +95. ¿Son grados, una lectura cruda del AS5600 u otra escala?

---

## 3. El dataset ⭐ *(indispensable)*

- [x] **Cómo se juntaron los datos** (parcial)
- [ ] **De dónde sale la etiqueta** ❓
- [x] **Cantidad y tipo de datos** (parcial)
- [x] **Aumentos de datos**

**Respuesta:**
- **La etiqueta sale del nombre del archivo.** El formato es `ID_STEERING_X_Y.jpg`; por ejemplo, `1000_-10_-14_0.jpg` tiene id = 1000 y steering_raw = −10.
  - ❓ No se sabe qué son el 3.er y el 4.º campo (por ejemplo `-14` y `0`). ¿Son otro sensor, el acelerador, un estado?
  - ❓ No se sabe si el ángulo viene del AS5600 a través del ESP32. El formato se parece al del servidor de telemetría (que guarda los datos en el nombre de la foto), pero no es igual.
- **Cantidad:**
  - 5351 imágenes válidas, en 4 carpetas (`imagenes001` a `imagenes004`).
  - Se descartó una carpeta `Corruptas`.
  - Distribución: media −0.058, desvío 0.20, mínimo −1.09, máximo 0.95. Hay muchas muestras cerca de 0 (manejo recto).
- **Balanceo:** histograma de 25 bins, con un máximo de 400 muestras por bin. Se eliminaron 3244 imágenes y quedaron **2107**.
- **División:** 80/20 → **1685 de entrenamiento / 422 de validación** (`train_test_split`, `random_state=6`).
  - ⚠️ La división es aleatoria por imagen. Imágenes consecutivas de la misma vuelta pueden quedar en entrenamiento y en validación a la vez, así que la validación puede resultar optimista.
- ❓ **Calles:** no se sabe qué calle corresponde a cada carpeta (bulevar o adoquines).
- **Aumentos de datos** (solo en entrenamiento, cada uno con probabilidad 50%):
  - desplazamiento (*pan*) de ±10% en x e y;
  - zoom de 1.0 a 1.3×;
  - brillo de 0.2 a 1.2× (en el canal V de HSV);
  - **espejado horizontal invirtiendo el signo del ángulo**.

---

## 4. El entrenamiento

- [x] **Framework**
- [x] **Dónde se entrenó**
- [x] **Configuración**
- [x] **Qué significa "a medio entrenar"**

**Respuesta:**
- **Framework:** TensorFlow / Keras, en **Google Colab**.
- **Pérdida:** MSE.
- **Optimizador:** Adam, `lr = 1e-4`.
- **Lote:** 64. Hay 26 pasos por época en entrenamiento y 6 en validación (con un generador que toma muestras al azar).
- **Dos rondas sobre el mismo modelo:**
  1. 20 épocas máximo, con `EarlyStopping(patience=3)`. Se cortó en la época 5.
  2. Siguió entrenando ese mismo modelo: 30 épocas máximo, con `EarlyStopping(patience=6)` y `ReduceLROnPlateau` (×0.5, patience 3). **Se cortó en la época 17 de 30** y restauró los pesos de la época 11.
- **"A medio entrenar":** el entrenamiento se cortó por *early stopping*. La pérdida de entrenamiento se mantuvo cerca de 0.10 y no bajó de forma clara.

---

## 5. Resultados

- [x] **Alguna métrica**
- [x] **Pruebas** (parcial)

**Respuesta:**
- **Mejor `val_loss` = 0.0755 (MSE)**, en la época 11 de la segunda ronda.
  - Eso equivale a un RMSE de ≈ 0.27 en la escala normalizada, o sea **unas 27 unidades de `steering_raw`**. Es un error alto comparado con el desvío de los datos (0.20).
  - La pérdida de validación osciló entre 0.075 y 0.11. Como se mide con solo 6 lotes al azar, es una métrica ruidosa.
- **Prueba sobre video** (`videotest.mp4`, 1139 frames, con un suavizado de 5 frames):
  - predicción mínima −0.29, máxima +0.15 y promedio −0.014;
  - el modelo predice giros chicos y casi siempre cerca de 0;
  - el video anotado se guardó como `videotest_predicho.mp4` en Drive.
- ❓ No hay registro de pruebas en el vehículo real.
- El gráfico de pérdida se guardó en Drive como `training_loss_plot.png`. Sirve para el informe.

---

## 6. Integración con el resto del sistema ⭐ *(indispensable)*

- [x] **En el bulevar** (en realidad, en cualquier calle)
- [x] **En los adoquines**
- [x] **Selección**

**Respuesta:** la integración está en `main_nav.py` + `pipeline/control.py`. **No hay un modo distinto según el tipo de calle.** Lo que cambia la fuente del giro es la **confianza geométrica** de cada frame:

| Confianza geométrica | Fuente del giro |
|---|---|
| ≥ 0.70 | 100% controlador geométrico |
| entre 0.30 y 0.70 | mezcla lineal: `b·geométrico + (1−b)·BC`, con `b = (conf − 0.3) / 0.4` |
| ≤ 0.30 | 100% BC |

- **Controlador geométrico** = `0.40·Geométrico + 0.35·PID + 0.25·Pure Pursuit`.
  - El PID usa Kp 0.60, Ki 0.05 y Kd 0.10, y tiene anti-windup.
  - También se calcula un controlador Stanley, pero solo se muestra: no entra en la mezcla.
- **Confianza geométrica** = `(0.60·franjas_válidas + 0.15·borde_izq_visible + 0.15·borde_der_visible + 0.10·polinomio_ok − 0.15·faltó_un_borde) × confianza_segmentación`.
- **Filtros de seguridad**, en este orden: saturación ±1 → limitador de cambio (máx. 0.12 por frame) → EMA (α = 0.35).
- **Alerta de divergencia:** si |geométrico − BC| > 0.45, se muestra un aviso. No cambia el comando.
- **Selección manual:** el BC se activa con `--steer model_best_val_loss.keras`. Sin ese parámetro el sistema funciona solo con geometría (y BC vale 0).
- **Adoquines:** ❓ en el código no hay nada específico. Si en los adoquines la segmentación es mala, la confianza baja y el BC toma el control de forma automática. Pero eso es una consecuencia del diseño, no una regla programada.

⚠️ **Este pipeline es distinto del de `tesis`** (el de 3 cámaras con fusión multicámara). Los dos no están conectados: `tesis/controller.py` es solo proporcional y no usa BC.

---

## Plan B: si el código no aparece

~~Presentarlo como línea de desarrollo en curso.~~ **Ya no hace falta: el código existe.** Lo que falta es aclarar los ❓, que son datos que solo puede responder quien grabó el dataset.

### Preguntas que quedan abiertas
1. ¿En qué unidades está `steering_raw`, y de qué sensor sale?
2. ¿Qué son el 3.er y el 4.º número del nombre de las imágenes?
3. ¿Qué calle tiene cada carpeta (`imagenes001` a `imagenes004`)?
4. ¿Se probó en el vehículo real?
