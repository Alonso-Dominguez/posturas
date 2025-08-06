import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import pyautogui
import time
import threading
from pynput import keyboard
import webbrowser
import subprocess
import platform

class YouTubeGestureControl:
    def __init__(self, model_path="gesture_model.pkl", scaler_path="gesture_scaler.pkl"):
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Cargar modelo y escalador
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("✅ Modelo y escalador cargados correctamente")
        except FileNotFoundError:
            print("❌ Error: No se encontraron los archivos del modelo")
            print("Primero ejecuta el entrenamiento del modelo")
            self.model = None
            self.scaler = None
        
        # Mapeo de gestos
        self.gestos = {
            -1: "NO_IDENTIFICADO",
            0: "CERRADO",
            1: "ABIERTO", 
            2: "PULGAR_ARRIBA",
            3: "PAZ",
            4: "APUNTAR"
        }
        
        # Acciones de YouTube
        self.youtube_actions = {
            -1: "🤷 Esperando...",
            0: "⏯️ PAUSAR/REANUDAR",
            1: "🔊 VOLUMEN ARRIBA",
            2: "👍 ME GUSTA",
            3: "⏩ ADELANTAR 10s", 
            4: "⏭️ SIGUIENTE VIDEO"
        }
        
        # Configuración
        self.confidence_threshold = 0.9
        self.min_confidence_for_action = 0.85
        self.action_cooldown = 2.0  # Segundos entre acciones
        self.last_action_time = 0
        
        # Buffer para suavizar predicciones
        self.prediction_buffer = deque(maxlen=7)
        self.current_gesture = None
        self.gesture_confidence = 0
        
        # Control de aplicación
        self.running = True
        self.paused = False
        
        # Configurar PyAutoGUI
        pyautogui.FAILSAFE = False  # Desactivar failsafe temporalmente
        pyautogui.PAUSE = 0.2
        
        # Dar tiempo para que el navegador esté en foco
        self.browser_focus_delay = 1.0
        
        # Inicializar listener de teclado
        self.setup_keyboard_listener()
    
    def setup_keyboard_listener(self):
        """Configura el listener para controles de teclado"""
        def on_press(key):
            try:
                if key.char == 'p':
                    self.paused = not self.paused
                    status = "PAUSADO" if self.paused else "ACTIVO"
                    print(f"🎮 Control gestual: {status}")
                elif key.char == 'q':
                    self.running = False
                    print("🛑 Cerrando aplicación...")
            except AttributeError:
                if key == keyboard.Key.esc:
                    self.running = False
                    print("🛑 Cerrando aplicación...")
        
        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()
    
    def open_youtube(self):
        """Abre YouTube en el navegador"""
        try:
            print("🌐 Abriendo YouTube...")
            webbrowser.open('https://www.youtube.com')
            time.sleep(4)  # Más tiempo para cargar
            print("✅ YouTube abierto.")
            print("🎯 IMPORTANTE: Haz clic en el video de YouTube para que esté en foco")
            print("   Las teclas solo funcionan cuando YouTube tiene el foco")
            input("   Presiona ENTER cuando hayas hecho clic en el video...")
        except Exception as e:
            print(f"❌ Error al abrir YouTube: {e}")
    
    def execute_youtube_action(self, gesture_id, confidence):
        """Ejecuta acciones específicas de YouTube"""
        current_time = time.time()
        
        # Verificar cooldown
        if current_time - self.last_action_time < self.action_cooldown:
            return "⏳ Cooldown activo"
        
        # Solo ejecutar si hay suficiente confianza y no está pausado
        if gesture_id == -1 or confidence < self.min_confidence_for_action or self.paused:
            return self.youtube_actions.get(gesture_id, "🤷 Sin acción")
        
        action_name = self.youtube_actions[gesture_id]
        
        try:
            print(f"\n🎯 Ejecutando acción: {action_name}")
            print(f"   Gesto ID: {gesture_id}, Confianza: {confidence:.2%}")
            
            # Pequeña pausa para asegurar que la ventana esté en foco
            time.sleep(0.1)
            
            if gesture_id == 0:  # CERRADO - Pausar/Reanudar
                print("   Enviando tecla: SPACE")
                pyautogui.press('space')
                time.sleep(0.2)
                print("   ✅ PAUSAR/REANUDAR ejecutado")
                
            elif gesture_id == 1:  # ABIERTO - Volumen arriba
                print("   Enviando tecla: UP")
                pyautogui.press('up')
                time.sleep(0.2)
                print("   ✅ VOLUMEN ARRIBA ejecutado")
                
            elif gesture_id == 2:  # PULGAR_ARRIBA - Me gusta
                print("   Enviando tecla: L")
                pyautogui.press('l')
                time.sleep(0.2)
                print("   ✅ ME GUSTA ejecutado")
                
            elif gesture_id == 3:  # PAZ - Adelantar 10 segundos
                print("   Enviando tecla: RIGHT")
                pyautogui.press('right')
                time.sleep(0.2)
                print("   ✅ ADELANTAR 10s ejecutado")
                
            elif gesture_id == 4:  # APUNTAR - Siguiente video
                print("   Enviando tecla: SHIFT+N")
                try:
                    pyautogui.hotkey('shift', 'n')
                    time.sleep(0.2)
                    print("   ✅ SIGUIENTE VIDEO ejecutado")
                except Exception as e:
                    print(f"   ⚠️ Hotkey falló, usando alternativa: {e}")
                    # Alternativa: mover hacia adelante mucho
                    pyautogui.press('right', presses=10, interval=0.1)
                    print("   ✅ AVANCE RÁPIDO ejecutado")
            
            self.last_action_time = current_time
            return f"✅ {action_name}"
            
        except Exception as e:
            print(f"   ❌ Error ejecutando acción: {e}")
            return f"❌ Error en {action_name}"
    
    def extract_landmarks(self, hand_landmarks):
        """Extrae coordenadas de landmarks - Arregla el warning de sklearn"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y])
        # Crear array con nombres de features consistentes
        landmarks_array = np.array(landmarks).reshape(1, -1)
        return landmarks_array
    
    def predict_with_confidence(self, landmarks):
        """Predice gesto con umbral de confianza"""
        if self.model is None or self.scaler is None:
            return -1, 0.0
        
        landmarks_scaled = self.scaler.transform(landmarks)
        probabilities = self.model.predict_proba(landmarks_scaled)[0]
        max_probability = np.max(probabilities)
        predicted_class = self.model.predict(landmarks_scaled)[0]
        
        if max_probability >= self.confidence_threshold:
            return predicted_class, max_probability
        else:
            return -1, max_probability
    
    def get_stable_prediction(self, prediction, confidence):
        """Suaviza las predicciones usando un buffer"""
        if confidence >= 0.7:
            self.prediction_buffer.append(prediction)
        
        if len(self.prediction_buffer) < 5:
            return None, 0.0
        
        predictions = list(self.prediction_buffer)
        most_common = max(set(predictions), key=predictions.count)
        stability = predictions.count(most_common) / len(predictions)
        
        if stability >= 0.7:
            return most_common, stability
        else:
            return -1, stability
    
    def draw_youtube_interface(self, frame, gesture_id, confidence, action_result, raw_confidence=0.0):
        """Dibuja interfaz específica para YouTube"""
        # Fondo para información
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Título
        cv2.putText(frame, "🎬 CONTROL GESTUAL DE YOUTUBE", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Estado
        status_color = (0, 255, 0) if not self.paused else (0, 165, 255)
        status_text = "ACTIVO" if not self.paused else "PAUSADO"
        cv2.putText(frame, f"Estado: {status_text}", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Gesto actual
        gesture_name = self.gestos.get(gesture_id, "DESCONOCIDO")
        color = (0, 255, 0) if gesture_id != -1 else (0, 165, 255)
        cv2.putText(frame, f"Gesto: {gesture_name}", 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Acción ejecutada
        cv2.putText(frame, f"Accion: {action_result}", 
                   (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Métricas
        cv2.putText(frame, f"Confianza: {raw_confidence:.1%} | Estabilidad: {confidence:.1%}", 
                   (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Cooldown
        cooldown_remaining = max(0, self.action_cooldown - (time.time() - self.last_action_time))
        if cooldown_remaining > 0:
            cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s", 
                       (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        
        # Controles de YouTube en el lado derecho
        youtube_controls = [
            "✊ CERRADO → ⏯️ Pausar/Reanudar",
            "✋ ABIERTO → 🔊 Subir Volumen", 
            "👍 PULGAR → 👍 Me Gusta",
            "✌️ PAZ → ⏩ Adelantar 10s",
            "👉 APUNTAR → ⏭️ Siguiente"
        ]
        
        cv2.putText(frame, "CONTROLES:", 
                   (650, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, control in enumerate(youtube_controls):
            y_pos = 60 + (i * 25)
            cv2.putText(frame, control, 
                       (650, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Barra de confianza
        bar_width = int(300 * raw_confidence)
        bar_color = (0, 255, 0) if raw_confidence >= self.confidence_threshold else (0, 0, 255)
        cv2.rectangle(frame, (650, 180), (650 + bar_width, 200), bar_color, -1)
        cv2.rectangle(frame, (650, 180), (950, 200), (255, 255, 255), 2)
        
        # Línea del umbral
        threshold_x = int(650 + 300 * self.confidence_threshold)
        cv2.line(frame, (threshold_x, 175), (threshold_x, 205), (255, 255, 0), 2)
        
        # Instrucciones en la parte inferior
        instructions = [
            "P: Pausar/Reactivar control | Q/ESC: Salir | +/-: Ajustar umbral",
            "¡Asegurate de tener YouTube abierto en tu navegador!"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = frame.shape[0] - 40 + (i * 20)
            cv2.putText(frame, instruction, 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_youtube_control(self):
        """Ejecuta el control de YouTube"""
        if self.model is None:
            print("❌ Error: Modelo no cargado")
            return
        
        # Preguntar si abrir YouTube
        response = input("¿Deseas abrir YouTube automáticamente? (y/n): ").strip().lower()
        if response in ['y', 'yes', 'si', 's']:
            self.open_youtube()
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n🎬 === CONTROL GESTUAL DE YOUTUBE INICIADO ===")
        print("\n📋 CONTROLES DISPONIBLES:")
        for gesture_id, action in self.youtube_actions.items():
            if gesture_id != -1:
                gesture_name = self.gestos[gesture_id]
                print(f"   {gesture_name} → {action}")
        
        print(f"\n⚙️ CONFIGURACIÓN:")
        print(f"   • Umbral de confianza: {self.confidence_threshold:.0%}")
        print(f"   • Cooldown entre acciones: {self.action_cooldown}s")
        print(f"   • Umbral para acciones: {self.min_confidence_for_action:.0%}")
        
        print(f"\n🎮 CONTROLES DE TECLADO:")
        print(f"   • P: Pausar/reactivar control gestual")
        print(f"   • Q/ESC: Salir de la aplicación")
        print(f"   • +/-: Ajustar umbral de confianza")
        
        print(f"\n📱 IMPORTANTE PARA QUE FUNCIONE:")
        print(f"   1. Abre YouTube en tu navegador")
        print(f"   2. Reproduce cualquier video")
        print(f"   3. HAZ CLIC EN EL VIDEO (debe tener foco)")
        print(f"   4. Mantén la ventana del navegador visible")
        print(f"   5. Los gestos deben ser claros y mantenidos por 1-2 segundos")
        
        print(f"\n⚠️  TROUBLESHOOTING:")
        print(f"   • Si no funciona: haz clic en el video de YouTube")
        print(f"   • Mantén la ventana de YouTube visible (no minimizada)")
        print(f"   • Prueba presionar SPACE manualmente para confirmar que funciona")
        
        current_action_result = "Esperando gesto..."
        
        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks and not self.paused:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extraer y predecir
                    landmarks = self.extract_landmarks(hand_landmarks)
                    prediction, raw_confidence = self.predict_with_confidence(landmarks)
                    
                    # Suavizar predicción
                    stable_result = self.get_stable_prediction(prediction, raw_confidence)
                    
                    if stable_result[0] is not None:
                        stable_gesture, stability = stable_result
                        
                        if stability >= 0.7:
                            if self.current_gesture != stable_gesture:
                                self.current_gesture = stable_gesture
                                current_action_result = self.execute_youtube_action(
                                    stable_gesture, raw_confidence)
                            
                            self.gesture_confidence = stability
            
            elif self.paused:
                current_action_result = "⏸️ Control pausado (presiona P para reactivar)"
            else:
                cv2.putText(frame, "👋 Coloca tu mano frente a la camara", 
                           (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.prediction_buffer.clear()
                self.current_gesture = None
                current_action_result = "Esperando mano..."
            
            # Dibujar interfaz
            if hasattr(self, 'current_gesture') and self.current_gesture is not None:
                self.draw_youtube_interface(
                    frame, self.current_gesture, 
                    self.gesture_confidence, current_action_result, 
                    raw_confidence if 'raw_confidence' in locals() else 0.0
                )
            else:
                self.draw_youtube_interface(frame, -1, 0.0, current_action_result, 0.0)
            
            cv2.imshow('🎬 Control Gestual de YouTube', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = min(0.99, self.confidence_threshold + 0.05)
                print(f"📈 Umbral aumentado a: {self.confidence_threshold:.0%}")
            elif key == ord('-'):
                self.confidence_threshold = max(0.5, self.confidence_threshold - 0.05)
                print(f"📉 Umbral disminuido a: {self.confidence_threshold:.0%}")
            elif key == ord('p'):
                self.paused = not self.paused
                status = "PAUSADO" if self.paused else "ACTIVO"
                print(f"🎮 Control gestual: {status}")
        
        cap.release()
        cv2.destroyAllWindows()
        self.listener.stop()
        print("🏁 Control de YouTube finalizado")

def main():
    print("🎬 Iniciando Control Gestual de YouTube...")
    
    # Verificar dependencias
    try:
        import pyautogui
        import pynput
        print("✅ Todas las dependencias están instaladas")
    except ImportError as e:
        print(f"❌ Falta dependencia: {e}")
        print("Instala con: pip install pyautogui pynput")
        return
    
    controller = YouTubeGestureControl()
    controller.run_youtube_control()

if __name__ == "__main__":
    main()