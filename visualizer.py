"""
visualizer.py — Visualización de debug del pipeline de navegación.

Genera una ventana única con toda la información necesaria para
entender y depurar el sistema en tiempo real.
"""

import time
from typing import Dict, Optional

import cv2
import numpy as np

import config
from nav_types import (
    CameraPosition, CameraResult, FusionResult, NavCommand, SystemState
)


class DebugVisualizer:
    """
    Genera visualización de debug del pipeline completo.

    Layout:
    ┌────────────────────────────────────────────────┐
    │  [LEFT + overlay]  [FRONT + overlay]  [RIGHT]  │
    │   Q=0.72 W=0.30    Q=0.91 W=0.60    Q=0.45   │
    ├────────────────────────────────────────────────┤
    │  FUSION: ahead=+0.12  heading=5.3°             │
    │  CMD: lin_x=0.35  ang_z=-0.12  STATE: NORMAL   │
    │  FPS: 14.2  Cameras: 3/3                       │
    └────────────────────────────────────────────────┘
    """

    def __init__(self):
        self.fps_counter = _FPSCounter()
        self.window_name = "AGV Navigation Debug"

    def render(
        self,
        camera_results: Dict[CameraPosition, CameraResult],
        fusion: FusionResult,
        nav_cmd: NavCommand,
        detection=None
    ) -> np.ndarray:
        """
        Genera y muestra la visualización de debug.

        Args:
            camera_results: Resultados por cámara.
            fusion: Resultado de la fusión.
            nav_cmd: Comando de navegación actual.

        Returns:
            La imagen del canvas completo (para grabación o screenshot).
        """
        if not config.VIS_ENABLED:
            return np.zeros((10, 10, 3), dtype=np.uint8)

        self.fps_counter.tick()

        # Tamaño de cada panel de cámara
        panel_w = config.CAMERA_WIDTH
        panel_h = config.CAMERA_HEIGHT

        # Generar panel de cada cámara
        panels = {}
        cam_order = [CameraPosition.LEFT, CameraPosition.FRONT, CameraPosition.RIGHT]

        for pos in cam_order:
            result = camera_results.get(pos)
            det = detection if pos == CameraPosition.FRONT else None
            if result is not None:
                panels[pos] = self._render_camera_panel(result, fusion, det)
            else:
                panels[pos] = self._blank_panel(panel_w, panel_h, pos.value)

        # Ensamblar fila de cámaras
        cam_row = np.hstack([panels[pos] for pos in cam_order])

        # Panel inferior con información de fusión y control
        info_panel = self._render_info_panel(
            cam_row.shape[1], fusion, nav_cmd, camera_results
        )

        # Ensamblar canvas final
        canvas = np.vstack([cam_row, info_panel])

        # Aplicar escala
        if config.VIS_SCALE != 1.0:
            new_w = int(canvas.shape[1] * config.VIS_SCALE)
            new_h = int(canvas.shape[0] * config.VIS_SCALE)
            canvas = cv2.resize(canvas, (new_w, new_h))

        # Mostrar
        cv2.imshow(self.window_name, canvas)

        return canvas

    def _render_camera_panel(
        self, result: CameraResult, fusion: FusionResult, detection=None
    ) -> np.ndarray:
        """Renderiza el panel de una cámara con overlay de máscara y trayectoria."""

        # Frame base (o frame negro si no hay frame)
        if result.frame is not None:
            panel = result.frame.copy()
        else:
            panel = np.zeros(
                (config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8
            )

        h, w = panel.shape[:2]

        road_bool = (
            result.clean_mask > 127
            if result.clean_mask is not None and result.mask.valid
            else None
        )

        # --- Overlay segmentacion 2cams (naranja) — solo fuera del camino ---
        if detection is not None and detection.seg_overlay is not None:
            seg = detection.seg_overlay.copy()
            if road_bool is not None:
                seg[road_bool] = 0   # Borrar donde ya hay camino verde
            panel = cv2.addWeighted(panel, 1.0, seg, 0.4, 0)

        # --- Overlay de máscara camino (verde) ---
        if road_bool is not None:
            mask_overlay = np.zeros_like(panel)
            mask_overlay[road_bool] = config.VIS_MASK_COLOR
            panel = cv2.addWeighted(
                panel, 1.0, mask_overlay, config.VIS_MASK_ALPHA, 0
            )

        # --- Bounding boxes de deteccion (yolov8n + 2cams detect) ---
        if detection is not None:
            for label, conf, (x1, y1, x2, y2), color in detection.boxes:
                cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
                tag = f"{label} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(panel, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(panel, tag, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # --- Centerline ---
        if result.trajectory.valid and len(result.trajectory.centerline) > 1:
            pts = [(int(p.x), int(p.y)) for p in result.trajectory.centerline]
            for i in range(len(pts) - 1):
                cv2.line(panel, pts[i], pts[i + 1],
                         config.VIS_CENTERLINE_COLOR, 2)
            # Dibujar puntos individuales
            for pt in pts:
                cv2.circle(panel, pt, 3, config.VIS_CENTERLINE_COLOR, -1)

        # --- Ahead point de esta cámara ---
        if result.trajectory.ahead_point is not None:
            ax, ay = int(result.trajectory.ahead_point[0]), int(result.trajectory.ahead_point[1])
            cv2.circle(panel, (ax, ay), 8, config.VIS_AHEAD_POINT_COLOR, -1)
            cv2.circle(panel, (ax, ay), 10, config.VIS_AHEAD_POINT_COLOR, 2)

        # --- Ahead point fusionado (solo en cámara frontal) ---
        if result.camera_id == CameraPosition.FRONT and fusion.valid:
            fused_x = int((fusion.ahead_point_norm + 1.0) / 2.0 * w)
            fused_y = int(h * 0.5)  # Mostrar a mitad de altura como referencia
            cv2.circle(panel, (fused_x, fused_y), 12,
                       config.VIS_FUSED_POINT_COLOR, 3)
            cv2.putText(panel, "FUSED", (fused_x - 20, fused_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        config.VIS_FUSED_POINT_COLOR, 1)

        # --- Texto de info ---
        cam_name = result.camera_id.value.upper()
        quality = result.quality.total
        weight = fusion.effective_weights.get(result.camera_id.value, 0.0)

        # Nombre de cámara
        cv2.putText(panel, cam_name, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    config.VIS_TEXT_COLOR, 2)

        # [TEMP captura] Q y W grandes con borde negro (sin caja)
        for text, y, color in ((f"Q={quality:.2f}", h - 62, (0, 255, 0)),
                               (f"W={weight:.2f}", h - 12, (0, 255, 255))):
            cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (0, 0, 0), 10)
            cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, color, 4)

        # Heading angle
        if result.trajectory.valid:
            cv2.putText(
                panel,
                f"H={result.trajectory.heading_angle:.1f}deg",
                (w - 140, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                config.VIS_TEXT_COLOR, 1
            )

        # Ahead point normalizado
        if result.trajectory.valid:
            cv2.putText(
                panel,
                f"A={result.trajectory.ahead_point_norm:+.2f}",
                (w - 140, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                config.VIS_TEXT_COLOR, 1
            )

        # Borde si la cámara no es usable
        if not result.is_usable:
            cv2.rectangle(panel, (0, 0), (w - 1, h - 1),
                          (0, 0, 200), 3)
            cv2.putText(panel, "NO DATA", (w // 2 - 50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 200), 2)

        return panel

    def _blank_panel(self, w: int, h: int, label: str) -> np.ndarray:
        """Genera un panel negro con el nombre de la cámara."""
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(panel, label.upper(), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    config.VIS_TEXT_COLOR, 2)
        cv2.putText(panel, "DISCONNECTED", (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 200), 2)
        cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (0, 0, 200), 3)
        return panel

    def _render_info_panel(
        self,
        width: int,
        fusion: FusionResult,
        nav_cmd: NavCommand,
        camera_results: Dict[CameraPosition, CameraResult]
    ) -> np.ndarray:
        """Renderiza el panel inferior con información de fusión y control."""

        panel_h = 100
        panel = np.zeros((panel_h, width, 3), dtype=np.uint8)

        # Fondo ligeramente diferente
        panel[:] = (30, 30, 30)

        # Línea separadora superior
        cv2.line(panel, (0, 0), (width, 0), (80, 80, 80), 2)

        y_base = 25
        x_margin = 15

        # Fila 1: Fusión
        fusion_text = (
            f"FUSION: ahead={fusion.ahead_point_norm:+.3f}  "
            f"heading={fusion.heading_angle:+.1f}deg  "
            f"quality={fusion.global_quality:.2f}  "
            f"cameras={fusion.num_valid_cameras}/3"
        )
        cv2.putText(panel, fusion_text, (x_margin, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    config.VIS_TEXT_COLOR, 1)

        # Fila 2: Comando de navegación
        state_color = {
            SystemState.NORMAL: (0, 200, 0),
            SystemState.LOW_CONFIDENCE: (0, 180, 255),
            SystemState.STOPPED: (0, 0, 255),
            SystemState.NO_CAMERAS: (0, 0, 255),
        }.get(nav_cmd.state, (200, 200, 200))

        cmd_text = (
            f"CMD: linear.x={nav_cmd.linear_x:+.3f}  "
            f"angular.z={nav_cmd.angular_z:+.3f}"
        )
        cv2.putText(panel, cmd_text, (x_margin, y_base + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 255, 200), 1)

        # Estado
        state_text = f"STATE: {nav_cmd.state.value}"
        cv2.putText(panel, state_text, (x_margin + 450, y_base + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    state_color, 2)

        # Fila 3: FPS y pesos
        fps = self.fps_counter.fps
        weights_str = "  ".join(
            f"{k}={v:.2f}" for k, v in fusion.effective_weights.items()
        )
        fps_text = f"FPS: {fps:.1f}  |  Weights: {weights_str}"
        cv2.putText(panel, fps_text, (x_margin, y_base + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1)

        # [TEMP captura] Quality global grande a la derecha
        q_text = f"QUALITY={fusion.global_quality:.2f}"
        (tw, _), _ = cv2.getTextSize(q_text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)
        cv2.putText(panel, q_text, (width - tw - 20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 4)

        # Barra visual de steering
        self._draw_steering_bar(panel, width, nav_cmd)

        return panel

    def _draw_steering_bar(
        self, panel: np.ndarray, width: int, nav_cmd: NavCommand
    ):
        """Dibuja una barra visual de steering en el panel inferior."""
        bar_y = 85
        bar_h = 8
        bar_left = 15
        bar_right = width - 15
        bar_center = (bar_left + bar_right) // 2
        bar_width = bar_right - bar_left

        # Fondo de la barra
        cv2.rectangle(panel, (bar_left, bar_y),
                      (bar_right, bar_y + bar_h), (60, 60, 60), -1)
        # Centro
        cv2.line(panel, (bar_center, bar_y - 2),
                 (bar_center, bar_y + bar_h + 2), (120, 120, 120), 1)

        # Indicador de steering
        # angular.z negativo → giro derecha → indicador a la derecha
        steer_pos = int(bar_center - nav_cmd.angular_z / config.CONTROL_MAX_ANGULAR_Z * (bar_width // 2))
        steer_pos = max(bar_left, min(bar_right, steer_pos))

        indicator_color = (0, 200, 0) if nav_cmd.state == SystemState.NORMAL else (0, 0, 200)
        cv2.circle(panel, (steer_pos, bar_y + bar_h // 2), 6,
                   indicator_color, -1)

    def _quality_color(self, quality: float) -> tuple:
        """Devuelve color BGR según el nivel de calidad."""
        if quality > 0.7:
            return (0, 200, 0)      # Verde
        elif quality > 0.4:
            return (0, 200, 255)    # Amarillo/naranja
        else:
            return (0, 0, 255)      # Rojo

    def close(self):
        """Cierra la ventana de debug."""
        cv2.destroyAllWindows()


class _FPSCounter:
    """Contador de FPS con media móvil."""

    def __init__(self, window: int = 30):
        self._times = []
        self._window = window
        self.fps = 0.0

    def tick(self):
        now = time.time()
        self._times.append(now)
        # Mantener solo los últimos N timestamps
        if len(self._times) > self._window:
            self._times = self._times[-self._window:]
        if len(self._times) >= 2:
            elapsed = self._times[-1] - self._times[0]
            if elapsed > 0:
                self.fps = (len(self._times) - 1) / elapsed
