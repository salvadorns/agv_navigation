"""
main.py — Entry point del módulo de navegación visual AGV.

Pipeline principal:
1. Leer 3 cámaras
2. Inferencia YOLO por cámara
3. Postproceso de máscara
4. Extracción de trayectoria
5. Evaluación de calidad
6. Fusión multicámara
7. Control de navegación
8. Visualización de debug

Uso:
    python main.py                      # Modo cámaras en vivo
    python main.py --detect-cameras     # Detectar cámaras disponibles
    python main.py --video left.mp4 front.mp4 right.mp4  # Modo video
    python main.py --video l.mp4 f.mp4 r.mp4 --cuadros 984 \\
        --log runs/x.csv --debug-video runs/x.mp4         # Corrida registrada
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, Optional

import cv2

import config
from nav_types import CameraPosition, CameraResult, MaskResult, TrajectoryResult, QualityScore
from camera_manager import CameraManager, detect_available_cameras
from inference import SegmentationInference
from mask_processor import MaskProcessor
from trajectory import TrajectoryExtractor
from quality import QualityEvaluator
from fusion import MultiCameraFusion
from controller import NavigationController
from visualizer import DebugVisualizer
from detector import MultiDetector


class AGVNavigationPipeline:
    """
    Pipeline completo de navegación visual para el AGV.

    Orquesta todos los módulos y ejecuta el loop principal.
    Cada componente es independiente y reemplazable.
    """

    def __init__(
        self,
        video_sources: Optional[Dict[str, str]] = None,
        delay: int = 1,
        max_frames: Optional[int] = None,
        log_path: Optional[str] = None,
        debug_video_path: Optional[str] = None,
        debug_fps: float = 4.85,
    ):
        """
        Inicializa todos los componentes del pipeline.

        Args:
            video_sources: Si se pasan rutas de video, usar archivos
                           en vez de cámaras. Dict {"left": path, "front": path, "right": path}
            delay: Retraso en milisegundos entre frames (útil para ver videos en cámara lenta)
            max_frames: Si se indica, detener el loop tras procesar esa cantidad de frames.
            log_path: Si se indica, guardar un CSV con una fila por frame.
            debug_video_path: Si se indica, grabar la ventana de debug en un .mp4.
            debug_fps: FPS del video de debug grabado.
        """
        self.delay = delay
        self.max_frames = max_frames
        self.log_path = log_path
        self.debug_video_path = debug_video_path
        self.debug_fps = debug_fps
        self._log_rows = []
        self._video_writer = None
        print("=" * 60)
        print("  AGV Visual Navigation Pipeline — V1")
        print("=" * 60)

        # --- Inicializar componentes ---
        print("\n[Init] Cargando modelo de segmentación...")
        self.inference = SegmentationInference()

        print("[Init] Inicializando postprocesador de máscaras...")
        self.mask_processor = MaskProcessor()

        print("[Init] Inicializando extractor de trayectoria...")
        self.trajectory_extractor = TrajectoryExtractor()

        print("[Init] Inicializando evaluador de calidad...")
        self.quality_evaluator = QualityEvaluator()

        print("[Init] Inicializando fusión multicámara...")
        self.fusion = MultiCameraFusion()

        print("[Init] Inicializando controlador de navegación...")
        self.controller = NavigationController()

        print("[Init] Inicializando visualizador de debug...")
        self.visualizer = DebugVisualizer()

        print("[Init] Cargando detector de objetos (yolov8n + 2cams)...")
        self.detector = MultiDetector()

        # --- Inicializar cámaras ---
        if video_sources:
            print(f"[Init] Modo VIDEO: {video_sources}")
            self.camera_manager = self._setup_video_sources(video_sources)
        else:
            print("[Init] Modo CÁMARA en vivo...")
            self.camera_manager = CameraManager()

        self.running = False
        print("\n[Init] Pipeline listo.\n")

    def _setup_video_sources(
        self, video_sources: Dict[str, str]
    ) -> CameraManager:
        """Configura CameraManager para usar archivos de video en lugar de cámaras."""
        # Crear un CameraManager con índices de video
        # Mapear los paths a "índices" que OpenCV acepta como strings
        video_indices = {}
        for name, path in video_sources.items():
            video_indices[name] = path  # OpenCV acepta strings como fuente

        # Nota: CameraManager espera índices int, así que usamos un approach
        # ligeramente diferente para videos
        manager = _VideoCameraManager(video_sources)
        return manager

    def run(self):
        """
        Ejecuta el loop principal del pipeline.

        El loop corre hasta que se presione 'q' o Ctrl+C.
        """
        self.running = True
        print("[Pipeline] Iniciando loop principal. Presionar 'q' para salir.\n")

        frame_count = 0

        try:
            while self.running:
                if self.max_frames is not None and frame_count >= self.max_frames:
                    print(f"\n[Pipeline] Se alcanzaron {self.max_frames} frames.")
                    break

                frame_count += 1

                # ==============================================================
                # PASO 1: Leer cámaras
                # ==============================================================
                frames = self.camera_manager.read_all()

                # ==============================================================
                # PASO 2-5: Procesar cada cámara
                # ==============================================================
                camera_results: Dict[CameraPosition, CameraResult] = {}

                for pos in CameraPosition:
                    cam_frame = frames[pos]

                    if not cam_frame.valid:
                        # Cámara no disponible — resultado vacío
                        camera_results[pos] = CameraResult(
                            camera_id=pos,
                            frame=None,
                            mask=MaskResult(),
                            clean_mask=None,
                            trajectory=TrajectoryResult(
                                centerline=[], ahead_point=None, valid=False,
                                num_valid_slices=0
                            ),
                            quality=QualityScore()
                        )
                        continue

                    frame = cam_frame.frame

                    # Paso 2: Inferencia YOLO
                    mask_result = self.inference.run(frame)

                    # Paso 3: Postproceso de máscara
                    clean_mask = None
                    if mask_result.valid and mask_result.mask is not None:
                        clean_mask = self.mask_processor.process(mask_result.mask)
                        # Si la máscara limpia está vacía, marcar como inválida
                        if clean_mask is not None and cv2.countNonZero(clean_mask) == 0:
                            mask_result = MaskResult(
                                mask=clean_mask, confidence=mask_result.confidence,
                                valid=False, class_id=mask_result.class_id
                            )
                    else:
                        clean_mask = None

                    # Paso 4: Extracción de trayectoria
                    if clean_mask is not None and mask_result.valid:
                        traj_result = self.trajectory_extractor.extract(clean_mask)
                    else:
                        traj_result = TrajectoryResult(
                            centerline=[], ahead_point=None, valid=False,
                            num_valid_slices=0
                        )

                    # Paso 5: Evaluación de calidad
                    quality_score = self.quality_evaluator.compute(
                        pos, mask_result, traj_result, clean_mask
                    )

                    camera_results[pos] = CameraResult(
                        camera_id=pos,
                        frame=frame,
                        mask=mask_result,
                        clean_mask=clean_mask,
                        trajectory=traj_result,
                        quality=quality_score
                    )

                # ==============================================================
                # PASO 6: Fusión multicámara
                # ==============================================================
                fusion_result = self.fusion.fuse(camera_results)

                # ==============================================================
                # PASO 7: Control de navegación
                # ==============================================================
                nav_cmd = self.controller.compute(fusion_result)

                # ==============================================================
                # PASO 8: Salida y visualización
                # ==============================================================

                if self.log_path:
                    self._log_rows.append(
                        self._log_row(frame_count - 1, camera_results, fusion_result, nav_cmd)
                    )

                # Imprimir comando cada N frames para no saturar la consola
                if frame_count % 10 == 0:
                    print(f"[F{frame_count:05d}] {nav_cmd}")

                # Deteccion de objetos sobre la camara frontal
                front_frame = frames[CameraPosition.FRONT].frame if frames[CameraPosition.FRONT].valid else None
                detection = self.detector.run(front_frame)

                # Visualización de debug
                canvas = self.visualizer.render(camera_results, fusion_result, nav_cmd, detection)
                if self.debug_video_path:
                    self._write_debug_frame(canvas)

                # Esperar tecla (delay para controlar velocidad, 1ms=rápido, 100ms=lento)
                key = cv2.waitKey(self.delay) & 0xFF
                if key == ord('q') or key == 27:  # 'q' o ESC
                    print("\n[Pipeline] Salida solicitada por el usuario.")
                    self.running = False

        except KeyboardInterrupt:
            print("\n[Pipeline] Interrumpido por Ctrl+C.")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Libera todos los recursos."""
        print("[Pipeline] Liberando recursos...")
        self.camera_manager.release_all()
        self.visualizer.close()
        if self._video_writer is not None:
            self._video_writer.release()
            print(f"[Pipeline] Video de debug guardado -> {self.debug_video_path}")
        if self.log_path:
            self._save_log()
        print("[Pipeline] Finalizado.")

    # ==========================================================================
    # Registro de la corrida (CSV y video de debug)
    # ==========================================================================

    LOG_HEADER = [
        "cuadro", "estado", "linear_x", "angular_z", "calidad_global", "camaras_validas",
        "peso_izq", "peso_frontal", "peso_der", "q_izq", "q_frontal", "q_der",
        "ahead_fusionado", "heading_fusionado",
    ]

    @staticmethod
    def _log_row(frame_idx, camera_results, fusion, nav_cmd) -> list:
        """Arma la fila del CSV para un frame."""
        weights = fusion.effective_weights
        return [
            frame_idx, nav_cmd.state.value,
            f"{nav_cmd.linear_x:.4f}", f"{nav_cmd.angular_z:.4f}",
            f"{fusion.global_quality:.4f}", fusion.num_valid_cameras,
            *(f"{weights.get(c, 0.0):.3f}" for c in ("left", "front", "right")),
            *(f"{camera_results[p].quality.total:.4f}"
              for p in (CameraPosition.LEFT, CameraPosition.FRONT, CameraPosition.RIGHT)),
            f"{fusion.ahead_point_norm:.4f}", f"{fusion.heading_angle:.2f}",
        ]

    def _write_debug_frame(self, canvas):
        """Escribe el canvas de debug en el video (se crea con el primer frame)."""
        # Con la visualización desactivada render() devuelve un canvas vacío de 10x10
        if canvas is None or canvas.shape[0] <= 10:
            return
        if self._video_writer is None:
            _ensure_parent_dir(self.debug_video_path)
            h, w = canvas.shape[:2]
            self._video_writer = cv2.VideoWriter(
                self.debug_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                self.debug_fps, (w, h)
            )
        self._video_writer.write(canvas)

    def _save_log(self):
        """Escribe el CSV con todas las filas registradas."""
        _ensure_parent_dir(self.log_path)
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.LOG_HEADER)
            writer.writerows(self._log_rows)
        print(f"[Pipeline] {len(self._log_rows)} frames registrados -> {self.log_path}")

    def process_single_frame(
        self, frames: Dict[CameraPosition, any]
    ) -> dict:
        """
        Procesa un solo ciclo del pipeline.

        Útil para integración con ROS2 o testing.
        Devuelve un dict con todos los resultados intermedios.

        Args:
            frames: Dict de CameraFrame por posición.

        Returns:
            Dict con camera_results, fusion_result, nav_cmd.
        """
        camera_results = {}
        for pos in CameraPosition:
            cam_frame = frames.get(pos)
            if cam_frame is None or not cam_frame.valid:
                camera_results[pos] = CameraResult(
                    camera_id=pos, frame=None, mask=MaskResult(),
                    clean_mask=None,
                    trajectory=TrajectoryResult(
                        centerline=[], ahead_point=None, valid=False,
                        num_valid_slices=0
                    ),
                    quality=QualityScore()
                )
                continue

            frame = cam_frame.frame
            mask_result = self.inference.run(frame)
            clean_mask = None
            if mask_result.valid and mask_result.mask is not None:
                clean_mask = self.mask_processor.process(mask_result.mask)
                if clean_mask is not None and cv2.countNonZero(clean_mask) == 0:
                    mask_result = MaskResult(valid=False)

            traj_result = (
                self.trajectory_extractor.extract(clean_mask)
                if clean_mask is not None and mask_result.valid
                else TrajectoryResult(
                    centerline=[], ahead_point=None, valid=False, num_valid_slices=0
                )
            )
            quality_score = self.quality_evaluator.compute(
                pos, mask_result, traj_result, clean_mask
            )
            camera_results[pos] = CameraResult(
                camera_id=pos, frame=frame, mask=mask_result,
                clean_mask=clean_mask, trajectory=traj_result,
                quality=quality_score
            )

        fusion_result = self.fusion.fuse(camera_results)
        nav_cmd = self.controller.compute(fusion_result)

        return {
            "camera_results": camera_results,
            "fusion_result": fusion_result,
            "nav_cmd": nav_cmd,
        }


class _VideoCameraManager:
    """
    CameraManager alternativo que lee de archivos de video.

    Mismo interfaz que CameraManager para intercambiabilidad.
    """

    def __init__(self, video_sources: Dict[str, str]):
        from nav_types import CameraFrame

        self.captures: Dict[CameraPosition, Optional[cv2.VideoCapture]] = {}
        position_map = {
            "left": CameraPosition.LEFT,
            "front": CameraPosition.FRONT,
            "right": CameraPosition.RIGHT,
        }

        for name, path in video_sources.items():
            pos = position_map[name]
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                self.captures[pos] = cap
                print(f"[VideoManager] {pos.value}: {path} — OK")
            else:
                self.captures[pos] = None
                print(f"[VideoManager] {pos.value}: {path} — ERROR al abrir")

    def read_all(self):
        from nav_types import CameraFrame
        import time

        frames = {}
        now = time.time()

        for pos in CameraPosition:
            cap = self.captures.get(pos)
            if cap is None or not cap.isOpened():
                frames[pos] = CameraFrame(
                    frame=None, camera_id=pos, timestamp=now, valid=False
                )
                continue

            ret, frame = cap.read()
            if ret and frame is not None:
                # Redimensionar si es necesario
                h, w = frame.shape[:2]
                if w != config.CAMERA_WIDTH or h != config.CAMERA_HEIGHT:
                    frame = cv2.resize(
                        frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
                    )
                frames[pos] = CameraFrame(
                    frame=frame, camera_id=pos, timestamp=now, valid=True
                )
            else:
                # Video terminó — reiniciar desde el inicio (loop)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    if w != config.CAMERA_WIDTH or h != config.CAMERA_HEIGHT:
                        frame = cv2.resize(
                            frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
                        )
                    frames[pos] = CameraFrame(
                        frame=frame, camera_id=pos, timestamp=now, valid=True
                    )
                else:
                    frames[pos] = CameraFrame(
                        frame=None, camera_id=pos, timestamp=now, valid=False
                    )

        return frames

    def get_status(self):
        return {
            pos.value: (cap is not None and cap.isOpened())
            for pos, cap in self.captures.items()
        }

    def release_all(self):
        for pos, cap in self.captures.items():
            if cap is not None:
                cap.release()
                print(f"[VideoManager] {pos.value}: liberado")


def _ensure_parent_dir(path: str):
    """Crea la carpeta contenedora de un archivo si no existe."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AGV Visual Navigation Pipeline V1",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--detect-cameras",
        action="store_true",
        help="Detectar cámaras USB disponibles y salir."
    )

    parser.add_argument(
        "--video",
        nargs=3,
        metavar=("LEFT", "FRONT", "RIGHT"),
        help="Usar archivos de video en lugar de cámaras.\n"
             "Ejemplo: --video left.mp4 front.mp4 right.mp4"
    )

    parser.add_argument(
        "--front-video",
        type=str,
        metavar="VIDEO",
        help="Usar un solo video como cámara frontal.\n"
             "Las cámaras laterales quedan como no disponibles.\n"
             "Ejemplo: --front-video video_frontal_webcam.mp4"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Ruta al modelo .pt (default: {config.MODEL_PATH})"
    )

    parser.add_argument(
        "--cameras",
        nargs=3,
        type=int,
        metavar=("LEFT_IDX", "FRONT_IDX", "RIGHT_IDX"),
        help="Índices de cámaras USB.\n"
             "Ejemplo: --cameras 0 1 2"
    )

    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Desactivar visualización (modo headless)."
    )

    parser.add_argument(
        "--delay",
        type=int,
        default=1,
        help="Retraso en milisegundos entre frames. Aumentar para cámara lenta (ej: 100)."
    )

    parser.add_argument(
        "--cuadros",
        type=int,
        default=None,
        metavar="N",
        help="Detenerse después de procesar N frames\n"
             "(sin esto, los videos se repiten en loop)."
    )

    parser.add_argument(
        "--log",
        type=str,
        default=None,
        metavar="CSV",
        help="Guardar un CSV con una fila por frame (estado, comandos,\n"
             "calidad, pesos y calidad de cada cámara)."
    )

    parser.add_argument(
        "--debug-video",
        type=str,
        default=None,
        metavar="MP4",
        help="Grabar la ventana de debug en un video .mp4\n"
             "(no funciona con --no-vis)."
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=4.85,
        help="FPS del video de debug grabado (default: 4.85)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Modo detección de cámaras
    if args.detect_cameras:
        print("Detectando cámaras disponibles...")
        available = detect_available_cameras()
        if available:
            print(f"Cámaras encontradas en índices: {available}")
        else:
            print("No se encontraron cámaras.")
        return

    # Configurar overrides
    if args.model:
        config.MODEL_PATH = args.model

    if args.cameras:
        config.CAMERA_INDICES = {
            "left": args.cameras[0],
            "front": args.cameras[1],
            "right": args.cameras[2],
        }

    if args.no_vis:
        config.VIS_ENABLED = False

    # Crear y ejecutar pipeline
    video_sources = None
    if args.front_video:
        # Solo cámara frontal desde video
        video_sources = {"front": args.front_video}
    elif args.video:
        video_sources = {
            "left": args.video[0],
            "front": args.video[1],
            "right": args.video[2],
        }

    if args.debug_video and args.no_vis:
        print("[WARN] --debug-video no graba nada con --no-vis (no se dibuja la ventana).")

    pipeline = AGVNavigationPipeline(
        video_sources=video_sources,
        delay=args.delay,
        max_frames=args.cuadros,
        log_path=args.log,
        debug_video_path=args.debug_video,
        debug_fps=args.fps,
    )
    pipeline.run()


if __name__ == "__main__":
    main()

