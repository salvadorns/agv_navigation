"""
mask_processor.py — Postprocesamiento de máscaras de segmentación.

Limpia la máscara cruda de YOLO para obtener una representación robusta
del área navegable. Pipeline: binarización → cierre morfológico →
relleno de huecos → eliminación de blobs pequeños → componente principal.
"""

import cv2
import numpy as np

import config


class MaskProcessor:
    """
    Aplica postproceso a las máscaras de segmentación.

    Cada paso del pipeline está implementado como método separado
    para facilitar la depuración y el ajuste individual.
    """

    def __init__(self):
        # Pre-computar kernels morfológicos (se reutilizan en cada frame)
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.MASK_MORPH_KERNEL_SIZE, config.MASK_MORPH_KERNEL_SIZE)
        )
        self.fill_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.MASK_FILL_KERNEL_SIZE, config.MASK_FILL_KERNEL_SIZE)
        )

    def process(self, mask: np.ndarray) -> np.ndarray:
        """
        Pipeline completo de limpieza de máscara.

        Args:
            mask: Máscara binaria uint8 [0, 255].

        Returns:
            Máscara limpia uint8 [0, 255] con solo el componente principal.
        """
        if mask is None or mask.size == 0:
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH), dtype=np.uint8)

        # Asegurar formato correcto
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # 0. Recortar zona superior (cielo, edificios, horizonte — no son calle)
        crop_y = int(mask.shape[0] * config.MASK_TOP_CROP_RATIO)
        mask[:crop_y, :] = 0

        # 1. Binarización (por si la máscara tiene valores intermedios)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 2. Cierre morfológico — une regiones cercanas y cierra gaps pequeños
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.close_kernel)

        # 3. Relleno de huecos internos (cierre agresivo)
        filled = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, self.fill_kernel)

        # Relleno matemático de agujeros internos grandes usando contornos
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

        # 4. Extrusión hacia abajo: El modelo YOLO a veces recorta la máscara a la mitad
        # debido a cómo fue etiquetado el dataset. Para que la calle llegue hasta el fondo:
        filled = self._extrude_bottom(filled)

        # 5. Eliminación de blobs pequeños
        result = self._keep_large_components(filled)

        return result

    def _extrude_bottom(self, mask: np.ndarray) -> np.ndarray:
        """
        Extiende la parte más baja detectada de la máscara hacia el fondo de la imagen.
        Resuelve el problema del modelo entrenado con polígonos que no llegan al borde inferior.
        """
        h, w = mask.shape
        
        # Encontrar las filas que tienen píxeles blancos
        rows_with_pixels = np.any(mask > 0, axis=1)
        
        # Encontrar la fila más baja (mayor índice Y) con píxeles
        lowest_y = -1
        for y in range(h - 1, -1, -1):
            if rows_with_pixels[y]:
                lowest_y = y
                break
                
        # Si la máscara ya llega hasta muy abajo (menos de 10px del borde), no hacer nada
        if lowest_y == -1 or lowest_y >= h - 10:
            return mask
            
        # Si la región más baja detectada está muy arriba, tal vez no es calle, evitamos un falso positivo ciego
        if lowest_y < h // 2:
            return mask
            
        # Obtener los puntos izquierdo y derecho máximos en esa fila más baja
        row_pixels = np.where(mask[lowest_y] > 0)[0]
        if len(row_pixels) == 0:
            return mask
            
        x_min = row_pixels[0]
        x_max = row_pixels[-1]
        
        # Opcional para asegurar que se expande hacia afuera en perspectiva
        # Para ser seguros, simplemente dibujamos un rectángulo desde lowest_y hasta el fondo h
        result = mask.copy()
        cv2.rectangle(result, (x_min, lowest_y), (x_max, h), 255, -1)
        
        return result

    def _keep_large_components(self, mask: np.ndarray) -> np.ndarray:
        """Conserva solo el componente conectado más grande (la calle principal)."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        if num_labels <= 1:
            return np.zeros_like(mask)

        # Buscar el componente más grande (ignorando el fondo, label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_area  = stats[largest_label, cv2.CC_STAT_AREA]

        if largest_area < config.MASK_MIN_BLOB_AREA:
            return np.zeros_like(mask)

        result = np.zeros_like(mask)
        result[labels == largest_label] = 255
        return result
