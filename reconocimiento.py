import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

class GestureRecognitionRealTime:
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
            print("Modelo y escalador cargados correctamente")
        except FileNotFoundError:
            print("Error: No se encontraron los archivos del modelo")
            print("Primero ejecuta el entrenamiento del modelo")
            self.model = None
            self.scaler = None
        
        # Mapeo de gestos (agregamos NO_IDENTIFICADO)
        self.gestos = {
            -1: "NO_IDENTIFICADO",
            0: "CERRADO",
            1: "ABIERTO", 
            2: "PULGAR_ARRIBA",
            3: "PAZ",
            4: "APUNTAR"
        }
        
        # Acciones asociadas a cada gesto
        self.acciones = {
            -1: "🤷 SIN_ACCION",
            0: "⏸️ PAUSAR",
            1: "▶️ REPRODUCIR",
            2: "👍 ME_GUSTA",
            3: "✌️ COMPARTIR", 
            4: "👉 SIGUIENTE"
        }
        
        # Configuración de confianza
        self.confidence_threshold = 0.9  # 90% de confianza mínima
        self.min_confidence_for_action = 0.85  # Para ejecutar acciones
        
        # Buffer para suavizar predicciones
        self.prediction_buffer = deque(maxlen=5)
        self.current_gesture = None
        self.gesture_confidence = 0
        
    def extract_landmarks(self, hand_landmarks):
        """Extrae coordenadas de landmarks"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y])
        return np.array(landmarks).reshape(1, -1)
    
    def predict_with_confidence(self, landmarks):
        """Predice gesto con umbral de confianza"""
        if self.model is None or self.scaler is None:
            return -1, 0.0
        
        # Normalizar datos
        landmarks_scaled = self.scaler.transform(landmarks)
        
        # Obtener probabilidades
        probabilities = self.model.predict_proba(landmarks_scaled)[0]
        max_probability = np.max(probabilities)
        predicted_class = self.model.predict(landmarks_scaled)[0]
        
        # Aplicar umbral de confianza
        if max_probability >= self.confidence_threshold:
            return predicted_class, max_probability
        else:
            return -1, max_probability  # NO_IDENTIFICADO
    
    def get_stable_prediction(self, prediction, confidence):
        """Suaviza las predicciones usando un buffer"""
        # Solo agregar al buffer si la confianza es razonable
        if confidence >= 0.7:  # Umbral más bajo para el buffer
            self.prediction_buffer.append(prediction)
        
        if len(self.prediction_buffer) < 3:
            return None, 0.0
        
        # Obtener la predicción más frecuente
        predictions = list(self.prediction_buffer)
        most_common = max(set(predictions), key=predictions.count)
        stability = predictions.count(most_common) / len(predictions)
        
        # Solo devolver predicción estable si tiene suficiente consenso
        if stability >= 0.6:
            return most_common, stability
        else:
            return -1, stability  # NO_IDENTIFICADO si no hay consenso
    
    def draw_gesture_info(self, frame, gesture_id, confidence, action, raw_confidence=0.0):
        """Dibuja información del gesto en pantalla"""
        # Fondo para el texto
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Texto del gesto con color según confianza
        gesture_name = self.gestos.get(gesture_id, "DESCONOCIDO")
        color = (0, 255, 0) if gesture_id != -1 else (0, 165, 255)  # Verde o naranja
        
        cv2.putText(frame, f"Gesto: {gesture_name}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Acción
        action_color = (255, 255, 0) if gesture_id != -1 else (128, 128, 128)
        cv2.putText(frame, f"Accion: {action}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2)
        
        # Confianza del modelo (cruda)
        cv2.putText(frame, f"Confianza modelo: {raw_confidence:.1%}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Estabilidad de la predicción
        cv2.putText(frame, f"Estabilidad: {confidence:.1%}", 
                   (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Umbral visual
        threshold_text = f"Umbral: {self.confidence_threshold:.0%}"
        cv2.putText(frame, threshold_text, 
                   (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Barra de confianza del modelo
        bar_width = int(200 * raw_confidence)
        bar_color = (0, 255, 0) if raw_confidence >= self.confidence_threshold else (0, 0, 255)
        cv2.rectangle(frame, (520, 30), (520 + bar_width, 50), bar_color, -1)
        cv2.rectangle(frame, (520, 30), (720, 50), (255, 255, 255), 2)
        
        # Línea del umbral
        threshold_x = int(520 + 200 * self.confidence_threshold)
        cv2.line(frame, (threshold_x, 25), (threshold_x, 55), (255, 255, 0), 2)
        
        # Barra de estabilidad
        stability_width = int(200 * confidence)
        cv2.rectangle(frame, (520, 70), (520 + stability_width, 90), (255, 165, 0), -1)
        cv2.rectangle(frame, (520, 70), (720, 90), (255, 255, 255), 2)
    
    def simulate_action(self, gesture_id, confidence):
        """Simula ejecutar una acción basada en el gesto"""
        # Solo ejecutar acción si la confianza es muy alta
        if gesture_id == -1 or confidence < self.min_confidence_for_action:
            action = self.acciones.get(-1, "SIN_ACCION")
            print(f"🤷 Gesto no identificado con suficiente confianza ({confidence:.1%})")
        else:
            action = self.acciones.get(gesture_id, "SIN_ACCION")
            print(f"🎬 Ejecutando acción: {action} (Confianza: {confidence:.1%})")
        
        return action
    
    def run_recognition(self):
        """Ejecuta el reconocimiento en tiempo real"""
        if self.model is None:
            print("Error: Modelo no cargado")
            return
        
        cap = cv2.VideoCapture(0)
        
        print("=== RECONOCIMIENTO DE GESTOS CON UMBRAL DE CONFIANZA ===")
        print(f"Umbral de confianza: {self.confidence_threshold:.0%}")
        print(f"Umbral para acciones: {self.min_confidence_for_action:.0%}")
        print("Controles simulados:")
        for gesture_id, action in self.acciones.items():
            if gesture_id != -1:
                gesture_name = self.gestos[gesture_id]
                print(f"  {gesture_name} -> {action}")
        print(f"  {self.gestos[-1]} -> {self.acciones[-1]}")
        print("\nControles:")
        print("- Presiona 'q' para salir")
        print("- Presiona '+' para aumentar umbral")
        print("- Presiona '-' para disminuir umbral")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Voltear imagen
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe
            results = self.hands.process(rgb_frame)
            
            current_action = "Esperando gesto..."
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extraer características
                    landmarks = self.extract_landmarks(hand_landmarks)
                    
                    # Predecir con umbral de confianza
                    prediction, raw_confidence = self.predict_with_confidence(landmarks)
                    
                    # Suavizar predicción
                    stable_result = self.get_stable_prediction(prediction, raw_confidence)
                    
                    if stable_result[0] is not None:
                        stable_gesture, stability = stable_result
                        
                        # Solo cambiar si hay suficiente estabilidad
                        if stability >= 0.6:
                            if self.current_gesture != stable_gesture:
                                self.current_gesture = stable_gesture
                                current_action = self.simulate_action(stable_gesture, raw_confidence)
                            else:
                                current_action = self.acciones.get(stable_gesture, "SIN_ACCION")
                            
                            self.gesture_confidence = stability
                        
                        # Dibujar información (incluyendo confianza cruda del modelo)
                        if hasattr(self, 'current_gesture') and self.current_gesture is not None:
                            self.draw_gesture_info(
                                frame, self.current_gesture, 
                                self.gesture_confidence, current_action, raw_confidence
                            )
            else:
                # No hay mano detectada
                cv2.putText(frame, "Coloca tu mano frente a la camara", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.prediction_buffer.clear()
                self.current_gesture = None
            
            # Instrucciones
            cv2.putText(frame, "Q: Salir | +/-: Ajustar umbral", 
                       (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.putText(frame, f"Umbral actual: {self.confidence_threshold:.0%}", 
                       (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Reconocimiento de Gestos', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = min(0.99, self.confidence_threshold + 0.05)
                print(f"Umbral aumentado a: {self.confidence_threshold:.0%}")
            elif key == ord('-'):
                self.confidence_threshold = max(0.5, self.confidence_threshold - 0.05)
                print(f"Umbral disminuido a: {self.confidence_threshold:.0%}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Reconocimiento finalizado")

def main():
    recognizer = GestureRecognitionRealTime()
    recognizer.run_recognition()

if __name__ == "__main__":
    main()