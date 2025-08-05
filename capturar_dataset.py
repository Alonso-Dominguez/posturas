import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from datetime import datetime

class GestureDatasetCapture:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Estados/gestos que vamos a capturar
        self.gestos = {
            0: "cerrado",      # Puño cerrado
            1: "abierto",      # Mano abierta
            2: "pulgar_arriba", # Like
            3: "paz",          # Señal de paz
            4: "apuntar"       # Dedo índice apuntando
        }
        
        self.current_gesture = 0
        self.dataset_file = "gesture_dataset.csv"
        self.samples_per_gesture = 50  # Muestras por gesto
        self.current_samples = 0
        
        # Crear archivo CSV si no existe
        self.init_csv()
    
    def init_csv(self):
        """Inicializa el archivo CSV con headers"""
        if not os.path.exists(self.dataset_file):
            with open(self.dataset_file, 'w', newline='') as file:
                writer = csv.writer(file)
                # Header: 21 landmarks x 2 coordenadas (x,y) + estado
                headers = []
                for i in range(21):
                    headers.extend([f'x{i}', f'y{i}'])
                headers.append('estado')
                writer.writerow(headers)
    
    def extract_landmarks(self, hand_landmarks):
        """Extrae las coordenadas x,y de todos los landmarks"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y])
        return landmarks
    
    def save_sample(self, landmarks, gesture_id):
        """Guarda una muestra en el CSV"""
        with open(self.dataset_file, 'a', newline='') as file:
            writer = csv.writer(file)
            row = landmarks + [gesture_id]
            writer.writerow(row)
    
    def run_capture(self):
        """Ejecuta la captura de dataset"""
        cap = cv2.VideoCapture(0)
        
        print("=== CAPTURA DE DATASET PARA GESTOS ===")
        print("Instrucciones:")
        print("- Presiona ESPACIO para capturar muestra del gesto actual")
        print("- Presiona 'n' para cambiar al siguiente gesto")
        print("- Presiona 'q' para salir")
        print(f"- Se necesitan {self.samples_per_gesture} muestras por gesto")
        print("\nGestos a capturar:")
        for id, name in self.gestos.items():
            print(f"  {id}: {name}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Voltear imagen horizontalmente
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Dibujar landmarks si se detecta mano
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Información en pantalla
            current_gesture_name = self.gestos.get(self.current_gesture, "Desconocido")
            cv2.putText(frame, f"Gesto actual: {current_gesture_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Muestras capturadas: {self.current_samples}/{self.samples_per_gesture}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Instrucciones
            cv2.putText(frame, "ESPACIO: Capturar | N: Siguiente gesto | Q: Salir", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Captura de Dataset', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capturar muestra
            if key == ord(' '):
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        self.save_sample(landmarks, self.current_gesture)
                        self.current_samples += 1
                        print(f"Muestra {self.current_samples} capturada para gesto '{current_gesture_name}'")
                        
                        if self.current_samples >= self.samples_per_gesture:
                            print(f"¡Completadas todas las muestras para '{current_gesture_name}'!")
                            print("Presiona 'n' para el siguiente gesto")
                else:
                    print("No se detectó mano. Coloca tu mano frente a la cámara.")
            
            # Siguiente gesto
            elif key == ord('n'):
                if self.current_gesture < len(self.gestos) - 1:
                    self.current_gesture += 1
                    self.current_samples = 0
                    print(f"Cambiando a gesto: {self.gestos[self.current_gesture]}")
                else:
                    print("¡Dataset completo! Presiona 'q' para salir.")
            
            # Salir
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Dataset guardado en: {self.dataset_file}")

if __name__ == "__main__":
    # Verificar que MediaPipe esté instalado
    try:
        import mediapipe as mp
        print("MediaPipe detectado correctamente")
    except ImportError:
        print("Error: MediaPipe no está instalado")
        print("Instala con: pip install mediapipe opencv-python")
        exit()
    
    # Ejecutar captura
    capture = GestureDatasetCapture()
    capture.run_capture()