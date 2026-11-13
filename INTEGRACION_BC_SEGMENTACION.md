# Integración del modelo de Behavioral Cloning con la segmentación

> **Fuente:** `Desktop\autonomouse vehicle\` — archivos `main_nav.py`, `pipeline/control.py` y `pipeline/geometry.py`.
> Todo lo que sigue sale de leer el código. **No se ejecutó** para medir valores reales.

---

## Resumen

En el código **no hay un "modo bulevar" ni un "modo adoquines"**. Hay un solo mecanismo que funciona igual en cualquier calle: en cada cuadro, el sistema mide qué tan bien ve el camino y, según eso, decide cuánto confía en la geometría y cuánto en el modelo de behavioral cloning (BC).

**Lo que cambia entre el bulevar y los adoquines es la confianza, no el modo.**

---

## 1. Cómo funciona, paso a paso

### 1.1 Dos "conductores" en paralelo

En cada cuadro se calculan dos propuestas de giro:

| Conductor | Qué usa | Cómo decide el giro |
|---|---|---|
| **Geométrico** | La máscara de segmentación (YOLO) | Encuentra los bordes del camino y calcula el giro con PID + Pure Pursuit + control geométrico |
| **BC** | La imagen cruda (red PilotNet) | Predice el giro imitando a un conductor humano. **No usa la segmentación** |

El conductor geométrico combina tres estrategias:
```
giro_geométrico = 0.40 × Geométrica + 0.35 × PID + 0.25 × Pure Pursuit
```

### 1.2 Confianza en la geometría (de 0 a 1)

Se calcula en `geometry.py`, líneas 338 a 343:

```
confianza = ( 0.60 × proporción de líneas donde se encontró camino
            + 0.15 si se ve el borde izquierdo
            + 0.15 si se ve el borde derecho
            + 0.10 si se pudo ajustar la curva del camino
            − 0.15 si faltó un borde y hubo que estimarlo )
            × confianza del modelo de segmentación
```

Hay 8 líneas de escaneo horizontales, ubicadas entre el 25% y el 88% de la altura de la imagen.

### 1.3 Quién maneja según la confianza

| Confianza | Quién maneja |
|---|---|
| **≥ 0.70** | 100% geométrico |
| **entre 0.30 y 0.70** | mezcla de los dos |
| **≤ 0.30** | 100% BC |

En la zona intermedia la mezcla es lineal:
```
b = (confianza − 0.30) / 0.40
giro = b × geométrico + (1 − b) × BC
```

| Confianza | Geométrico | BC |
|---|---|---|
| 0.30 | 0% | 100% |
| 0.40 | 25% | 75% |
| 0.50 | 50% | 50% |
| 0.60 | 75% | 25% |
| 0.70 | 100% | 0% |

### 1.4 Filtros de seguridad

Se aplican siempre, en este orden:
1. **Saturación:** el giro se limita a ±1.
2. **Limitador de cambio:** el giro no puede variar más de 0.12 por cuadro.
3. **Suavizado EMA:** α = 0.35.

### 1.5 Alerta de divergencia

Si `|giro_geométrico − giro_BC| > 0.45`, el sistema muestra un aviso en pantalla. **No cambia el comando**: solo informa que los dos conductores no están de acuerdo.

---

## 2. Qué pasa en cada calle

> Esto es una **consecuencia del diseño**, no una regla programada.

### Bulevar
El asfalto liso tiene bordes claros, la segmentación funciona bien y la confianza suele quedar alta. Maneja casi siempre la geometría. El BC entra en la mezcla solo cuando algo tapa el camino o hay sombras fuertes.

### Adoquines
Si la textura confunde a la segmentación, la confianza baja y el **BC toma el control solo**. Pero el código no sabe que está en adoquines. Si la segmentación anduviera bien sobre adoquines, ahí también manejaría la geometría.

> ⚠️ **Pendiente de verificar:** no se midió la confianza real en ninguna de las dos calles. Para afirmar "en adoquines maneja el BC" hay que correr `main_nav.py` con un video de adoquines y mirar en la consola qué fuente aparece: `src=GEOM`, `src=BLEN` o `src=BC`.

---

## 3. Cómo se elige el modo

La única elección es **manual y se hace al arrancar el programa**:

| Opción | Comando | Comportamiento |
|---|---|---|
| **Con BC** | `--steer model_best_val_loss.keras` | Sistema completo, con la mezcla automática |
| **Sin BC** | se omite `--steer` | Solo geometría. Cuando la confianza baja, el BC vale 0 y el vehículo tiende a ir recto |

Ejemplo:
```
python main_nav.py --input video.mp4 --seg 2cams.pt --steer model_best_val_loss.keras --output output_nav.mp4
```

**No hay ninguna opción para elegir el tipo de calle.**

---

## 4. Cómo decirlo en la presentación

> "El sistema combina dos conductores: uno geométrico, que sigue los bordes del camino detectados por la segmentación, y uno aprendido, que imita a un humano. En cada cuadro mide qué tan bien ve el camino. Si lo ve claro, como en el bulevar, confía en la geometría. Si la detección se degrada, como puede pasar en calles adoquinadas, le da el control progresivamente al modelo aprendido. No hace falta elegir el modo a mano: el cambio es automático."

**Nota:** si la diapositiva dice "bulevar y calle adoquinada", esa frase es honesta siempre que se diga **"puede pasar"** y no "pasa", hasta que se mida.

---

## 5. Aclaración importante

Este mecanismo pertenece al proyecto **`autonomouse vehicle`** (una cámara). El pipeline de **3 cámaras** del repositorio `tesis` **no usa BC**: su controlador es proporcional y no está conectado con este sistema.
