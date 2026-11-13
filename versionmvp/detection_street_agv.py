import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import numpy as np

# --- Modelo YOLO ---
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# --- CLASE CONTROLADOR AGV SIMPLIFICADA ---
class SimpleAGVController:
    def __init__(self):
        self.kp = 0.6
        self.ki = 0.01
        self.kd = 0.08
        self.max_steering_angle = 25
        self.smoothing_window = 5
        
        # Variables PID
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.steering_history = []
        
        # Variables para suavizado avanzado
        self.alpha = 0.3  # Factor de filtro exponencial (0 = más suave, 1 = más reactivo)
        self.last_smooth_steering = 0.0
        self.velocity_limit = 5.0  # Máximo cambio por frame en grados
        
        print("✅ Controlador AGV inicializado correctamente")
    
    def reset(self):
        """Reinicia el controlador"""
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.steering_history = []
        self.last_smooth_steering = 0.0
        print("🔄 Controlador reiniciado")
    
    def smooth_steering(self, raw_steering):
        """Suaviza el comando de dirección con múltiples técnicas"""
        # 1. Filtro exponencial (suavizado principal)
        exponential_smooth = self.alpha * raw_steering + (1 - self.alpha) * self.last_smooth_steering
        
        # 2. Limitador de velocidad (evita cambios bruscos)
        max_change = self.velocity_limit
        change = exponential_smooth - self.last_smooth_steering
        
        if abs(change) > max_change:
            limited_steering = self.last_smooth_steering + np.sign(change) * max_change
        else:
            limited_steering = exponential_smooth
        
        # 3. Media móvil adicional para casos extremos
        self.steering_history.append(limited_steering)
        if len(self.steering_history) > self.smoothing_window:
            self.steering_history.pop(0)
        
        moving_average = np.mean(self.steering_history)
        
        # 4. Combinación final (70% limitado + 30% media móvil)
        final_steering = 0.7 * limited_steering + 0.3 * moving_average
        
        # Actualizar para siguiente iteración
        self.last_smooth_steering = final_steering
        
        return final_steering
    
    def extract_road_centerline_advanced(self, image, results):
        """Extrae línea central más robusta usando múltiples puntos"""
        try:
            if not results.boxes or len(results.boxes) == 0:
                return None, 0
            
            # Tomar todas las detecciones con buena confianza
            good_boxes = []
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > 0.6:  # Solo cajas muy confiables
                    good_boxes.append((box, conf))
            
            if not good_boxes:
                print("❌ No hay detecciones confiables")
                return None, 0
            
            # Ordenar por confianza
            good_boxes.sort(key=lambda x: x[1], reverse=True)
            
            # Usar las mejores detecciones
            centers_x = []
            for box, conf in good_boxes[:3]:  # Máximo 3 mejores
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center_x = (x1 + x2) // 2
                centers_x.append(center_x)
            
            # Calcular centro promedio ponderado
            if len(centers_x) == 1:
                final_center_x = centers_x[0]
            else:
                # Eliminar outliers
                centers_x = np.array(centers_x)
                median = np.median(centers_x)
                # Solo usar centros que estén cerca de la mediana
                valid_centers = centers_x[np.abs(centers_x - median) < 50]
                final_center_x = int(np.mean(valid_centers)) if len(valid_centers) > 0 else int(median)
            
            # Centro de la imagen
            img_center = image.shape[1] // 2
            lateral_error = final_center_x - img_center
            
            # Crear línea central
            h, w = image.shape[:2]
            centerline = [(final_center_x, h//4), (final_center_x, h//2), (final_center_x, 3*h//4)]
            
            print(f"📊 Boxes usadas: {len(centers_x)}, Centro final: {final_center_x}, Error: {lateral_error}")
            
            return centerline, lateral_error
            
        except Exception as e:
            print(f"❌ Error en extract_road_centerline_advanced: {e}")
            return None, 0
        """Extrae línea central usando las detecciones de YOLO"""
        try:
            if not results.boxes or len(results.boxes) == 0:
                print("❌ No hay detecciones")
                return None, 0
            
            # Tomar la detección con mayor confianza
            best_box = None
            best_conf = 0
            
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box
            
            if best_box is None or best_conf < 0.5:
                print(f"❌ Confianza muy baja: {best_conf}")
                return None, 0
            
            # Obtener coordenadas de la caja
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
            
            # Calcular centro de la detección
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Centro de la imagen
            img_center = image.shape[1] // 2
            
            # Error lateral (distancia del centro)
            lateral_error = center_x - img_center
            
            # DEBUGGING AVANZADO
            box_width = x2 - x1
            box_height = y2 - y1
            print(f"📊 Conf: {best_conf:.2f}, Box: {box_width}x{box_height}, Centro: {center_x}, Error: {lateral_error}")
            
            # Crear línea central simple
            h, w = image.shape[:2]
            centerline = [(center_x, y1), (center_x, center_y), (center_x, y2)]
            
            return centerline, lateral_error
            
        except Exception as e:
            print(f"❌ Error en extract_road_centerline: {e}")
            return None, 0
    
    def compute_steering(self, image, results):
        """Calcula el comando de steering"""
        try:
            # Usar algoritmo avanzado o simple según preferencia
            use_advanced = True  # Cambia a False para usar el simple
            
            if use_advanced:
                centerline, lateral_error = self.extract_road_centerline_advanced(image, results)
            else:
                centerline, lateral_error = self.extract_road_centerline(image, results)
            
            if centerline is None:
                return 0, None
            
            # Normalizar error
            img_width = image.shape[1]
            normalized_error = lateral_error / (img_width / 2)
            
            # Control PID simple
            proportional = self.kp * normalized_error
            
            self.integral_error += normalized_error
            integral = self.ki * self.integral_error
            
            derivative = self.kd * (normalized_error - self.previous_error)
            self.previous_error = normalized_error
            
            # Limitar integral para evitar windup
            if abs(self.integral_error) > 50:
                self.integral_error = 50 * np.sign(self.integral_error)
            
            # Salida PID
            pid_output = proportional + integral + derivative
            
            # Convertir a ángulo
            raw_steering_angle = np.clip(pid_output * self.max_steering_angle, 
                                       -self.max_steering_angle, 
                                       self.max_steering_angle)
            
            # Suavizar
            smooth_steering = self.smooth_steering(raw_steering_angle)
            
            print(f"🎯 Raw: {raw_steering_angle:.1f}°, Smooth: {smooth_steering:.1f}°, Error: {normalized_error:.2f}")
            
            return smooth_steering, centerline
            
        except Exception as e:
            print(f"❌ Error en compute_steering: {e}")
            import traceback
            traceback.print_exc()
            return 0, None

# --- INSTANCIA DEL CONTROLADOR ---
try:
    agv_controller = SimpleAGVController()
    print("✅ Controlador creado exitosamente")
except Exception as e:
    print(f"❌ Error creando controlador: {e}")
    agv_controller = None

# --- VARIABLES GLOBALES ---
agv_mode = False
steering_angle_var = None
last_steering_angle = 0

# --- FUNCIONES ---
def visualize_agv_control(image, centerline, steering_angle):
    """Visualiza el control del AGV"""
    if centerline is None:
        return image
    
    try:
        # Crear copia de la imagen
        overlay = image.copy()
        h, w = overlay.shape[:2]
        
        # Dibujar línea central (verde)
        for i in range(len(centerline) - 1):
            cv2.line(overlay, centerline[i], centerline[i+1], (0, 255, 0), 4)
        
        # Dibujar centro de imagen (azul)
        cv2.line(overlay, (w//2, 0), (w//2, h), (255, 0, 0), 2)
        
        # Dibujar dirección steering (rojo)
        center_bottom = (w//2, h - 60)
        steering_length = 80
        angle_rad = np.radians(steering_angle)
        steering_x = int(center_bottom[0] + steering_length * np.sin(angle_rad))
        steering_y = int(center_bottom[1] - steering_length * np.cos(angle_rad))
        
        cv2.arrowedLine(overlay, center_bottom, (steering_x, steering_y), (0, 0, 255), 5)
        
        # Textos informativos
        cv2.putText(overlay, f'Steering: {steering_angle:.1f}°', 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Velocidades de motor simuladas
        left_speed = 0.5 - (steering_angle / 30.0) * 0.3
        right_speed = 0.5 + (steering_angle / 30.0) * 0.3
        cv2.putText(overlay, f'Motor L: {left_speed:.2f} R: {right_speed:.2f}', 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return overlay
        
    except Exception as e:
        print(f"❌ Error en visualización: {e}")
        return image

def procesar_imagen(path):
    try:
        img = cv2.imread(path)
        if img is None:
            raise ValueError("No se pudo cargar la imagen")
        
        results = model(img)[0]
        
        if agv_mode and agv_controller is not None:
            # Procesar con control AGV
            steering_angle, centerline = agv_controller.compute_steering(img, results)
            
            # Actualizar GUI
            global last_steering_angle
            last_steering_angle = steering_angle
            if steering_angle_var:
                steering_angle_var.set(f"Steering: {steering_angle:.1f}°")
            
            # Crear visualización
            annotated = results.plot()
            annotated = visualize_agv_control(annotated, centerline, steering_angle)
        else:
            annotated = results.plot()
        
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"❌ Error procesando imagen: {e}")
        raise e

def seleccionar_imagen():
    ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
    if not ruta:
        return
    try:
        status_var.set("Procesando imagen...")
        img = procesar_imagen(ruta)
        mostrar_imagen(img)
        status_var.set("✅ Imagen procesada.")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo procesar la imagen.\n{str(e)}")
        status_var.set("❌ Error al procesar.")

def mostrar_imagen(img_cv2):
    img_pil = Image.fromarray(img_cv2)
    img_pil.thumbnail((640, 400))
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.config(image=img_tk)
    panel.image = img_tk

def deteccion_webcam():
    def run():
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "No se encontró una cámara.")
                return
            
            status_var.set("🔴 Webcam activa (presiona 'q' para cerrar)")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ No se pudo leer frame de webcam")
                    break
                
                results = model(frame)[0]
                
                if agv_mode and agv_controller is not None:
                    # Procesar con control AGV
                    steering_angle, centerline = agv_controller.compute_steering(frame, results)
                    
                    # Actualizar GUI
                    global last_steering_angle
                    last_steering_angle = steering_angle
                    if steering_angle_var:
                        root.after_idle(lambda: steering_angle_var.set(f"Steering: {steering_angle:.1f}°"))
                    
                    # Visualizar
                    annotated = results.plot()
                    annotated = visualize_agv_control(annotated, centerline, steering_angle)
                else:
                    annotated = results.plot()
                
                cv2.imshow("Webcam - Detección YOLO", annotated)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
        except Exception as e:
            print(f"❌ Error en webcam: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            status_var.set("Webcam cerrada.")
    
    threading.Thread(target=run, daemon=True).start()

def procesar_video():
    ruta = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")])
    if not ruta:
        return
        
    def run_video():
        try:
            cap = cv2.VideoCapture(ruta)
            if not cap.isOpened():
                messagebox.showerror("Error", "No se pudo abrir el video.")
                return
                
            status_var.set("▶ Procesando video...")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("✅ Video terminado")
                    break
                
                frame_count += 1
                
                results = model(frame)[0]
                
                if agv_mode and agv_controller is not None:
                    # Procesar con control AGV
                    steering_angle, centerline = agv_controller.compute_steering(frame, results)
                    
                    # Actualizar GUI cada 10 frames para no saturar
                    if frame_count % 10 == 0:
                        global last_steering_angle
                        last_steering_angle = steering_angle
                        if steering_angle_var:
                            root.after_idle(lambda: steering_angle_var.set(f"Steering: {steering_angle:.1f}°"))
                    
                    # Visualizar
                    annotated = results.plot()
                    annotated = visualize_agv_control(annotated, centerline, steering_angle)
                else:
                    annotated = results.plot()
                
                cv2.imshow("Video - Detección YOLO", annotated)
                
                # Controlar velocidad de reproducción
                if cv2.waitKey(50) & 0xFF == ord("q"):  # 50ms = ~20 FPS
                    break
                    
        except Exception as e:
            print(f"❌ Error en video: {e}")
            import traceback
            traceback.print_exc()
            status_var.set(f"❌ Error en video: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            status_var.set("✅ Video finalizado.")
    
    threading.Thread(target=run_video, daemon=True).start()

def toggle_agv_mode():
    global agv_mode
    agv_mode = not agv_mode
    
    if agv_mode:
        if agv_controller is not None:
            agv_controller.reset()
            btn_agv_toggle.config(text="🤖 Modo AGV: ON", bg="#10b981")
            agv_frame.pack(pady=10)
            status_var.set("🤖 Modo AGV activado")
        else:
            messagebox.showerror("Error", "Controlador AGV no disponible")
            agv_mode = False
    else:
        btn_agv_toggle.config(text="🤖 Modo AGV: OFF", bg="#6b7280")
        agv_frame.pack_forget()
        status_var.set("🟢 Modo detección normal")

def update_controller_params():
    if agv_controller is None:
        messagebox.showerror("Error", "Controlador no disponible")
        return
        
    try:
        # Parámetros PID
        agv_controller.kp = float(kp_var.get())
        agv_controller.ki = float(ki_var.get())
        agv_controller.kd = float(kd_var.get())
        agv_controller.max_steering_angle = float(max_angle_var.get())
        
        # Parámetros de suavizado
        agv_controller.alpha = float(alpha_var.get())
        agv_controller.velocity_limit = float(vel_limit_var.get())
        
        agv_controller.reset()
        status_var.set("✅ Parámetros actualizados")
    except ValueError:
        messagebox.showerror("Error", "Por favor, ingresa valores numéricos válidos")
    except Exception as e:
        messagebox.showerror("Error", f"Error actualizando parámetros: {str(e)}")

# --- GUI ---
root = tk.Tk()
root.title("YOLO Calle - Control AGV Final")
root.geometry("950x850")
root.configure(bg="#f9fafc")

# --- Logo ---
try:
    logo_img = Image.open("logo.png")
    logo_img.thumbnail((100, 100))
    logo_tk = ImageTk.PhotoImage(logo_img)
    logo_label = tk.Label(root, image=logo_tk, bg="#f9fafc")
    logo_label.image = logo_tk
    logo_label.pack(pady=(20, 5))
except:
    print("⚠️ No se pudo cargar el logo")

# --- Título ---
titulo = tk.Label(root, text="Detección Callejera + Control AGV", 
                  font=("Helvetica", 20, "bold"), bg="#f9fafc", fg="#111827")
titulo.pack(pady=10)

# --- Panel Imagen ---
panel = tk.Label(root, bg="#f9fafc")
panel.pack(pady=10)

# --- Botones principales ---
botones_frame = tk.Frame(root, bg="#f9fafc")
botones_frame.pack(pady=20)

btn_imagen = tk.Button(botones_frame, text="🖼️ Imagen", font=("Helvetica", 12), 
                      command=seleccionar_imagen, bg="#3b82f6", fg="white", 
                      padx=16, pady=8, relief="flat", cursor="hand2")
btn_imagen.pack(side="left", padx=10)

btn_webcam = tk.Button(botones_frame, text="📷 Webcam", font=("Helvetica", 12), 
                      command=deteccion_webcam, bg="#10b981", fg="white", 
                      padx=16, pady=8, relief="flat", cursor="hand2")
btn_webcam.pack(side="left", padx=10)

btn_video = tk.Button(botones_frame, text="🎥 Video", font=("Helvetica", 12), 
                     command=procesar_video, bg="#f97316", fg="white", 
                     padx=16, pady=8, relief="flat", cursor="hand2")
btn_video.pack(side="left", padx=10)

# --- Botón toggle AGV ---
btn_agv_toggle = tk.Button(botones_frame, text="🤖 Modo AGV: OFF", font=("Helvetica", 12), 
                          command=toggle_agv_mode, bg="#6b7280", fg="white", 
                          padx=16, pady=8, relief="flat", cursor="hand2")
btn_agv_toggle.pack(side="left", padx=10)

# --- Panel de control AGV ---
agv_frame = tk.Frame(root, bg="#f0f0f0", relief="ridge", bd=2)

agv_title = tk.Label(agv_frame, text="🤖 Control AGV", font=("Helvetica", 14, "bold"), 
                     bg="#f0f0f0", fg="#333")
agv_title.pack(pady=5)

# Información de steering
steering_angle_var = tk.StringVar(value="Steering: 0.0°")
steering_label = tk.Label(agv_frame, textvariable=steering_angle_var, 
                         font=("Helvetica", 12), bg="#f0f0f0", fg="#333")
steering_label.pack(pady=5)

# Parámetros PID
params_frame = tk.Frame(agv_frame, bg="#f0f0f0")
params_frame.pack(pady=10)

tk.Label(params_frame, text="Parámetros PID:", font=("Helvetica", 10, "bold"), 
         bg="#f0f0f0").grid(row=0, column=0, columnspan=4, pady=5)

tk.Label(params_frame, text="Kp:", bg="#f0f0f0").grid(row=1, column=0, padx=5)
kp_var = tk.StringVar(value="0.6")
tk.Entry(params_frame, textvariable=kp_var, width=8).grid(row=1, column=1, padx=5)

tk.Label(params_frame, text="Ki:", bg="#f0f0f0").grid(row=1, column=2, padx=5)
ki_var = tk.StringVar(value="0.01")
tk.Entry(params_frame, textvariable=ki_var, width=8).grid(row=1, column=3, padx=5)

tk.Label(params_frame, text="Kd:", bg="#f0f0f0").grid(row=2, column=0, padx=5)
kd_var = tk.StringVar(value="0.08")
tk.Entry(params_frame, textvariable=kd_var, width=8).grid(row=2, column=1, padx=5)

tk.Label(params_frame, text="Max Angle:", bg="#f0f0f0").grid(row=2, column=2, padx=5)
max_angle_var = tk.StringVar(value="25")
tk.Entry(params_frame, textvariable=max_angle_var, width=8).grid(row=2, column=3, padx=5)

# Parámetros de suavizado
smooth_frame = tk.Frame(agv_frame, bg="#f0f0f0")
smooth_frame.pack(pady=10)

tk.Label(smooth_frame, text="Suavizado:", font=("Helvetica", 10, "bold"), 
         bg="#f0f0f0").grid(row=0, column=0, columnspan=4, pady=5)

tk.Label(smooth_frame, text="Alpha:", bg="#f0f0f0").grid(row=1, column=0, padx=5)
alpha_var = tk.StringVar(value="0.3")
tk.Entry(smooth_frame, textvariable=alpha_var, width=8).grid(row=1, column=1, padx=5)

tk.Label(smooth_frame, text="Vel Limit:", bg="#f0f0f0").grid(row=1, column=2, padx=5)
vel_limit_var = tk.StringVar(value="5.0")
tk.Entry(smooth_frame, textvariable=vel_limit_var, width=8).grid(row=1, column=3, padx=5)

btn_update_params = tk.Button(agv_frame, text="Actualizar Parámetros", 
                             command=update_controller_params, bg="#8b5cf6", fg="white", 
                             padx=10, pady=5, relief="flat", cursor="hand2")
btn_update_params.pack(pady=10)

# --- Estado ---
status_var = tk.StringVar(value="🟢 Listo para usar")
estado = tk.Label(root, textvariable=status_var, font=("Helvetica", 11), 
                 bg="#f9fafc", fg="#6b7280")
estado.pack(pady=10)

# --- Footer ---
footer = tk.Label(root, text="Presioná 'q' para cerrar video o webcam | Algoritmo de suavizado avanzado", 
                 font=("Helvetica", 9), bg="#f9fafc", fg="#9ca3af")
footer.pack(pady=5)

print("🚀 Aplicación iniciada correctamente")
root.mainloop()

