"""
test_pipeline.py — Test sintético del pipeline de navegación.

Genera frames sintéticos con un camino dibujado y los pasa por todo
el pipeline para verificar que cada módulo funciona correctamente
SIN necesidad de cámaras físicas ni archivos de video.

Uso:
    py test_pipeline.py                  # Test completo con visualización
    py test_pipeline.py --no-vis         # Solo consola (sin ventana)
    py test_pipeline.py --scenario curve # Probar escenario de curva
"""

import argparse
import time
import sys
import math

import cv2
import numpy as np

import config
from nav_types import (
    CameraPosition, CameraFrame, CameraResult,
    MaskResult, TrajectoryResult, QualityScore
)
from inference import SegmentationInference
from mask_processor import MaskProcessor
from trajectory import TrajectoryExtractor
from quality import QualityEvaluator
from fusion import MultiCameraFusion
from controller import NavigationController
from visualizer import DebugVisualizer


# =============================================================================
# Generación de frames sintéticos
# =============================================================================

def generate_path_frame(
    width: int = 640,
    height: int = 480,
    path_center_x: float = 0.5,
    path_width_bottom: float = 0.6,
    path_width_top: float = 0.15,
    curve_offset: float = 0.0,
    noise_level: float = 10,
) -> np.ndarray:
    """
    Genera una imagen sintética que simula un camino visto desde una cámara.

    El camino es un trapecio con perspectiva (más ancho abajo, más angosto arriba)
    con una textura tipo tierra/asfalto simulada.

    Args:
        width, height: Dimensiones de la imagen.
        path_center_x: Centro horizontal del camino (0=izq, 1=der). 0.5 = centrado.
        path_width_bottom: Ancho del camino en la base (fracción del ancho de imagen).
        path_width_top: Ancho del camino arriba (fracción).
        curve_offset: Desplazamiento lateral del camino arriba (simula curva).
                      Negativo=curva izquierda, Positivo=curva derecha.
        noise_level: Nivel de ruido de fondo.

    Returns:
        Imagen BGR sintética con un camino visible.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Fondo: "terreno" gris oscuro con algo de textura
    frame[:] = (40, 50, 35)  # Pasto/tierra oscura
    noise = np.random.randint(0, noise_level, (height, width, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)

    # Cielo (parte superior)
    sky_height = int(height * 0.3)
    sky = np.zeros((sky_height, width, 3), dtype=np.uint8)
    for y in range(sky_height):
        ratio = y / sky_height
        sky[y, :] = (int(180 - 80 * ratio), int(140 - 40 * ratio), int(80 + 20 * ratio))
    frame[:sky_height] = sky

    # Dibujar el camino como un polígono (trapecio con perspectiva)
    center_px = int(path_center_x * width)
    curve_px = int(curve_offset * width)

    # Puntos del camino
    bottom_left = int(center_px - path_width_bottom * width / 2)
    bottom_right = int(center_px + path_width_bottom * width / 2)
    top_left = int(center_px + curve_px - path_width_top * width / 2)
    top_right = int(center_px + curve_px + path_width_top * width / 2)

    # Horizonte a ~35% desde arriba
    horizon_y = int(height * 0.35)

    path_pts = np.array([
        [bottom_left, height],
        [bottom_right, height],
        [top_right, horizon_y],
        [top_left, horizon_y],
    ], dtype=np.int32)

    # Color del camino (gris claro como asfalto/tierra)
    path_color = (120, 130, 115)
    cv2.fillPoly(frame, [path_pts], path_color)

    # Agregar líneas de borde del camino (más claro)
    cv2.line(frame, (bottom_left, height), (top_left, horizon_y), (180, 180, 160), 2)
    cv2.line(frame, (bottom_right, height), (top_right, horizon_y), (180, 180, 160), 2)

    # Línea central punteada
    for y in range(horizon_y, height, 20):
        ratio = (y - horizon_y) / (height - horizon_y)
        cx = int(center_px + curve_px * (1 - ratio))
        cv2.circle(frame, (cx, y), 2, (200, 200, 180), -1)

    return frame


def generate_scenario_frames(
    scenario: str = "straight",
    frame_idx: int = 0,
) -> dict:
    """
    Genera frames para las 3 cámaras según un escenario específico.

    Escenarios:
    - "straight": camino recto centrado
    - "curve_left": curva a la izquierda
    - "curve_right": curva a la derecha
    - "narrow": camino angosto
    - "lost_front": frontal sin camino (simula pérdida)
    - "all_lost": ninguna cámara detecta camino

    Returns:
        Dict {CameraPosition: np.ndarray} con los 3 frames.
    """
    w, h = config.CAMERA_WIDTH, config.CAMERA_HEIGHT

    # Animación suave para simular movimiento
    t = frame_idx * 0.02  # tiempo normalizado

    if scenario == "straight":
        # Camino recto, bien visible en las 3 cámaras
        left_frame = generate_path_frame(w, h, path_center_x=0.7, curve_offset=-0.05)
        front_frame = generate_path_frame(w, h, path_center_x=0.5, curve_offset=0.0)
        right_frame = generate_path_frame(w, h, path_center_x=0.3, curve_offset=0.05)

    elif scenario == "curve_left":
        # Curva a la izquierda: camino se desvía a la izquierda
        offset = -0.15 + 0.03 * math.sin(t)
        left_frame = generate_path_frame(w, h, path_center_x=0.6, curve_offset=offset - 0.1)
        front_frame = generate_path_frame(w, h, path_center_x=0.45, curve_offset=offset)
        right_frame = generate_path_frame(w, h, path_center_x=0.35, curve_offset=offset + 0.05)

    elif scenario == "curve_right":
        # Curva a la derecha: camino se desvía a la derecha
        offset = 0.15 + 0.03 * math.sin(t)
        left_frame = generate_path_frame(w, h, path_center_x=0.65, curve_offset=offset - 0.05)
        front_frame = generate_path_frame(w, h, path_center_x=0.55, curve_offset=offset)
        right_frame = generate_path_frame(w, h, path_center_x=0.4, curve_offset=offset + 0.1)

    elif scenario == "narrow":
        # Camino angosto (menor ancho)
        left_frame = generate_path_frame(w, h, path_center_x=0.7, path_width_bottom=0.3, path_width_top=0.08)
        front_frame = generate_path_frame(w, h, path_center_x=0.5, path_width_bottom=0.3, path_width_top=0.08)
        right_frame = generate_path_frame(w, h, path_center_x=0.3, path_width_bottom=0.3, path_width_top=0.08)

    elif scenario == "lost_front":
        # Frontal pierde el camino, solo laterales funcionan
        left_frame = generate_path_frame(w, h, path_center_x=0.7, curve_offset=-0.05)
        front_frame = np.random.randint(30, 60, (h, w, 3), dtype=np.uint8)  # Ruido
        right_frame = generate_path_frame(w, h, path_center_x=0.3, curve_offset=0.05)

    elif scenario == "all_lost":
        # Ninguna cámara ve camino
        left_frame = np.random.randint(30, 60, (h, w, 3), dtype=np.uint8)
        front_frame = np.random.randint(30, 60, (h, w, 3), dtype=np.uint8)
        right_frame = np.random.randint(30, 60, (h, w, 3), dtype=np.uint8)

    else:
        raise ValueError(f"Escenario desconocido: {scenario}")

    return {
        CameraPosition.LEFT: left_frame,
        CameraPosition.FRONT: front_frame,
        CameraPosition.RIGHT: right_frame,
    }


# =============================================================================
# Tests
# =============================================================================

def test_modules_individually():
    """
    Test individual de cada módulo con datos sintéticos.
    No requiere modelo YOLO — prueba postproceso, trayectoria, calidad, fusión, control.
    """
    print("=" * 60)
    print("  TEST: Módulos individuales (sin YOLO)")
    print("=" * 60)

    w, h = config.CAMERA_WIDTH, config.CAMERA_HEIGHT

    # --- 1. Generar una máscara sintética del camino ---
    print("\n[1] Generando máscara sintética del camino...")
    mask = np.zeros((h, w), dtype=np.uint8)
    # Dibujar un trapecio (camino)
    pts = np.array([
        [int(w * 0.2), h],
        [int(w * 0.8), h],
        [int(w * 0.55), int(h * 0.35)],
        [int(w * 0.45), int(h * 0.35)],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    print(f"   Máscara generada: {mask.shape}, área={np.sum(mask > 0)} px")

    # --- 2. Test MaskProcessor ---
    print("\n[2] Probando MaskProcessor...")
    mp = MaskProcessor()
    clean = mp.process(mask)
    area_clean = np.sum(clean > 0)
    print(f"   Máscara limpia: área={area_clean} px")
    assert area_clean > 0, "ERROR: MaskProcessor devolvió máscara vacía"
    print("   ✓ MaskProcessor OK")

    # --- 3. Test TrajectoryExtractor ---
    print("\n[3] Probando TrajectoryExtractor...")
    te = TrajectoryExtractor()
    traj = te.extract(clean)
    print(f"   Centerline points: {len(traj.centerline)}")
    print(f"   Ahead point: {traj.ahead_point}")
    print(f"   Ahead point norm: {traj.ahead_point_norm:.3f}")
    print(f"   Heading: {traj.heading_angle:.1f}°")
    print(f"   Valid: {traj.valid}")
    print(f"   Valid slices: {traj.num_valid_slices}/{config.TRAJECTORY_NUM_SLICES}")
    assert traj.valid, "ERROR: TrajectoryExtractor no generó trayectoria válida"
    assert len(traj.centerline) >= 2, "ERROR: Menos de 2 puntos en centerline"
    assert traj.ahead_point is not None, "ERROR: No hay ahead point"
    print("   ✓ TrajectoryExtractor OK")

    # --- 4. Test QualityEvaluator ---
    print("\n[4] Probando QualityEvaluator...")
    qe = QualityEvaluator()
    mask_result = MaskResult(mask=clean, confidence=0.85, valid=True, class_id=0)
    quality = qe.compute(CameraPosition.FRONT, mask_result, traj, clean)
    print(f"   Total: {quality.total:.3f}")
    print(f"   Área: {quality.area_score:.3f}")
    print(f"   Continuidad: {quality.continuity_score:.3f}")
    print(f"   Confianza: {quality.confidence_score:.3f}")
    print(f"   Consistencia: {quality.consistency_score:.3f}")
    print(f"   Estabilidad: {quality.stability_score:.3f}")
    assert quality.total > 0.0, "ERROR: QualityScore es 0"
    print("   ✓ QualityEvaluator OK")

    # --- 5. Test Fusión ---
    print("\n[5] Probando MultiCameraFusion...")
    fusion = MultiCameraFusion()

    # Crear resultados simulados para las 3 cámaras
    camera_results = {}
    for pos in CameraPosition:
        # Shift del camino según la posición de la cámara
        shift = {CameraPosition.LEFT: -0.2, CameraPosition.FRONT: 0.0, CameraPosition.RIGHT: 0.2}
        cam_mask = np.zeros((h, w), dtype=np.uint8)
        cx = 0.5 + shift[pos]
        pts_cam = np.array([
            [int(w * (cx - 0.3)), h],
            [int(w * (cx + 0.3)), h],
            [int(w * (cx + 0.05)), int(h * 0.35)],
            [int(w * (cx - 0.05)), int(h * 0.35)],
        ], dtype=np.int32)
        cv2.fillPoly(cam_mask, [pts_cam], 255)

        cam_traj = te.extract(cam_mask)
        cam_quality = qe.compute(pos, mask_result, cam_traj, cam_mask)

        camera_results[pos] = CameraResult(
            camera_id=pos,
            frame=np.zeros((h, w, 3), dtype=np.uint8),
            mask=MaskResult(mask=cam_mask, confidence=0.9, valid=True),
            clean_mask=cam_mask,
            trajectory=cam_traj,
            quality=cam_quality
        )

    fusion_result = fusion.fuse(camera_results)
    print(f"   Ahead fusionado: {fusion_result.ahead_point_norm:+.3f}")
    print(f"   Heading fusionado: {fusion_result.heading_angle:+.1f}°")
    print(f"   Calidad global: {fusion_result.global_quality:.3f}")
    print(f"   Cámaras válidas: {fusion_result.num_valid_cameras}/3")
    print(f"   Pesos: {fusion_result.effective_weights}")
    assert fusion_result.valid, "ERROR: Fusión no es válida"
    assert fusion_result.num_valid_cameras == 3, "ERROR: No hay 3 cámaras válidas"
    print("   ✓ MultiCameraFusion OK")

    # --- 6. Test Controller ---
    print("\n[6] Probando NavigationController...")
    ctrl = NavigationController()
    nav_cmd = ctrl.compute(fusion_result)
    print(f"   {nav_cmd}")
    print(f"   linear.x = {nav_cmd.linear_x:+.4f}")
    print(f"   angular.z = {nav_cmd.angular_z:+.4f}")
    print(f"   Estado: {nav_cmd.state.value}")
    assert nav_cmd.linear_x > 0, "ERROR: linear.x debería ser > 0"
    print("   ✓ NavigationController OK")

    print("\n" + "=" * 60)
    print("  ✓ TODOS LOS MÓDULOS PASARON EL TEST")
    print("=" * 60)


def test_full_pipeline_synthetic(scenario: str = "straight", num_frames: int = 100):
    """
    Test end-to-end con frames sintéticos pasados por YOLO.

    NOTA: Requiere el modelo best.pt. Si YOLO no detecta camino en los
    frames sintéticos (porque no fue entrenado con este tipo de imagen),
    el test igual valida que el pipeline no crashea y maneja la situación.
    """
    print("=" * 60)
    print(f"  TEST: Pipeline completo — Escenario '{scenario}'")
    print("=" * 60)

    # Inicializar componentes
    print("\n[Init] Cargando componentes...")
    try:
        inference = SegmentationInference()
    except Exception as e:
        print(f"   WARN: No se pudo cargar el modelo YOLO: {e}")
        print("   Ejecutando test sin inferencia YOLO (bypass)...")
        inference = None

    mask_proc = MaskProcessor()
    traj_ext = TrajectoryExtractor()
    quality_eval = QualityEvaluator()
    fusion = MultiCameraFusion()
    controller = NavigationController()
    visualizer = DebugVisualizer()

    print(f"\n[Test] Corriendo {num_frames} frames, escenario: {scenario}")
    print("[Test] Presionar 'q' para salir antes.\n")

    for frame_idx in range(num_frames):
        # Generar frames sintéticos
        synth_frames = generate_scenario_frames(scenario, frame_idx)

        camera_results = {}
        for pos in CameraPosition:
            frame = synth_frames[pos]

            # Intentar inferencia YOLO
            if inference is not None:
                mask_result = inference.run(frame)
            else:
                mask_result = MaskResult(valid=False)

            # Si YOLO no detecta nada (esperado con frames sintéticos),
            # usar la máscara del camino generada directamente
            if not mask_result.valid:
                # Fallback: generar máscara desde el frame
                # (detectar el camino por color, ya que lo generamos nosotros)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, synth_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
                # Limpiar con el mask processor
                clean = mask_proc.process(synth_mask)
                if clean is not None and np.sum(clean > 0) > config.MASK_MIN_BLOB_AREA:
                    mask_result = MaskResult(
                        mask=synth_mask, confidence=0.75, valid=True, class_id=0
                    )
                else:
                    clean = None
            else:
                clean = mask_proc.process(mask_result.mask) if mask_result.mask is not None else None

            # Trayectoria
            if clean is not None and mask_result.valid:
                traj = traj_ext.extract(clean)
            else:
                traj = TrajectoryResult(
                    centerline=[], ahead_point=None, valid=False, num_valid_slices=0
                )

            # Calidad
            quality = quality_eval.compute(pos, mask_result, traj, clean)

            camera_results[pos] = CameraResult(
                camera_id=pos,
                frame=frame,
                mask=mask_result,
                clean_mask=clean,
                trajectory=traj,
                quality=quality
            )

        # Fusión
        fusion_result = fusion.fuse(camera_results)

        # Control
        nav_cmd = controller.compute(fusion_result)

        # Imprimir cada 10 frames
        if frame_idx % 10 == 0:
            print(f"[F{frame_idx:04d}] {nav_cmd}  "
                  f"weights={fusion_result.effective_weights}")

        # Visualización
        if config.VIS_ENABLED:
            visualizer.render(camera_results, fusion_result, nav_cmd)
            key = cv2.waitKey(30) & 0xFF  # 30ms para ~33 FPS
            if key == ord('q') or key == 27:
                print("\n[Test] Salida solicitada por el usuario.")
                break

    visualizer.close()
    print(f"\n[Test] Completado. Último comando: {nav_cmd}")
    print(f"[Test] Estado final: {nav_cmd.state.value}")


def test_failover():
    """
    Test de resiliencia: simula pérdida progresiva de cámaras.
    """
    print("=" * 60)
    print("  TEST: Failover — Pérdida progresiva de cámaras")
    print("=" * 60)

    mask_proc = MaskProcessor()
    traj_ext = TrajectoryExtractor()
    quality_eval = QualityEvaluator()
    fusion = MultiCameraFusion()
    controller = NavigationController()

    w, h = config.CAMERA_WIDTH, config.CAMERA_HEIGHT

    def make_good_result(pos):
        """Crea un resultado válido para una cámara."""
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([
            [int(w * 0.2), h], [int(w * 0.8), h],
            [int(w * 0.55), int(h * 0.35)], [int(w * 0.45), int(h * 0.35)]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        traj = traj_ext.extract(mask)
        mr = MaskResult(mask=mask, confidence=0.9, valid=True)
        q = quality_eval.compute(pos, mr, traj, mask)
        return CameraResult(
            camera_id=pos, frame=np.zeros((h, w, 3), dtype=np.uint8),
            mask=mr, clean_mask=mask, trajectory=traj, quality=q
        )

    def make_dead_result(pos):
        """Crea un resultado inválido (cámara muerta)."""
        return CameraResult(
            camera_id=pos, frame=None, mask=MaskResult(),
            clean_mask=None,
            trajectory=TrajectoryResult(
                centerline=[], ahead_point=None, valid=False, num_valid_slices=0
            ),
            quality=QualityScore()
        )

    # Escenario 1: Todas funcionando
    print("\n[1] Todas las cámaras funcionando...")
    results = {pos: make_good_result(pos) for pos in CameraPosition}
    f = fusion.fuse(results)
    cmd = controller.compute(f)
    print(f"   {cmd}  cámaras={f.num_valid_cameras}/3")
    assert cmd.linear_x > 0, "ERROR: Debería avanzar"

    # Escenario 2: Cámara derecha muere
    print("\n[2] Cámara derecha muere...")
    results[CameraPosition.RIGHT] = make_dead_result(CameraPosition.RIGHT)
    f = fusion.fuse(results)
    cmd = controller.compute(f)
    print(f"   {cmd}  cámaras={f.num_valid_cameras}/3  pesos={f.effective_weights}")
    assert cmd.linear_x > 0, "ERROR: Debería seguir avanzando con 2 cámaras"
    assert "right" not in f.effective_weights or f.effective_weights.get("right", 0) == 0

    # Escenario 3: Solo queda la frontal
    print("\n[3] Solo cámara frontal...")
    results[CameraPosition.LEFT] = make_dead_result(CameraPosition.LEFT)
    f = fusion.fuse(results)
    cmd = controller.compute(f)
    print(f"   {cmd}  cámaras={f.num_valid_cameras}/3  pesos={f.effective_weights}")
    assert cmd.linear_x > 0, "ERROR: Debería avanzar con 1 cámara"

    # Escenario 4: Todas mueren
    print("\n[4] Todas las cámaras mueren...")
    results[CameraPosition.FRONT] = make_dead_result(CameraPosition.FRONT)
    f = fusion.fuse(results)
    cmd = controller.compute(f)
    print(f"   {cmd}  cámaras={f.num_valid_cameras}/3")
    # Después de algunos frames de decay, debería detenerse
    for _ in range(10):
        cmd = controller.compute(f)
    print(f"   Tras decay: {cmd}")
    assert cmd.linear_x == 0.0, "ERROR: Debería estar detenido"

    print("\n" + "=" * 60)
    print("  ✓ FAILOVER TEST PASADO")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test sintético del pipeline de navegación AGV",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--test", type=str, default="all",
        choices=["all", "modules", "pipeline", "failover"],
        help="Qué test correr:\n"
             "  all      — todos los tests\n"
             "  modules  — solo test de módulos individuales (sin YOLO)\n"
             "  pipeline — pipeline completo con frames sintéticos\n"
             "  failover — test de pérdida de cámaras"
    )
    parser.add_argument(
        "--scenario", type=str, default="straight",
        choices=["straight", "curve_left", "curve_right", "narrow", "lost_front", "all_lost"],
        help="Escenario para test de pipeline."
    )
    parser.add_argument(
        "--frames", type=int, default=150,
        help="Número de frames para test de pipeline."
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="Sin ventana de visualización."
    )

    args = parser.parse_args()

    if args.no_vis:
        config.VIS_ENABLED = False

    print("\n" + "🔧 " * 20)
    print("  AGV Navigation Pipeline — Test Suite")
    print("🔧 " * 20 + "\n")

    if args.test in ("all", "modules"):
        test_modules_individually()
        print()

    if args.test in ("all", "failover"):
        test_failover()
        print()

    if args.test in ("all", "pipeline"):
        test_full_pipeline_synthetic(args.scenario, args.frames)
        print()

    print("\n✅ Tests completados.")


if __name__ == "__main__":
    main()
