import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Función para detectar gestos básicos
def detectar_gesto(hand_landmarks):
    dedos_arriba = []

    # IDs de las puntas de los dedos
    tips = [4, 8, 12, 16, 20]

    for tip in tips:
        # Pulgar (x), los demás (y)
        if tip == 4:
            if hand_landmarks.landmark[tip].x < hand_landmarks.landmark[tip - 1].x:
                dedos_arriba.append(1)
            else:
                dedos_arriba.append(0)
        else:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                dedos_arriba.append(1)
            else:
                dedos_arriba.append(0)

    # Gestos simples basados en los dedos arriba
    if dedos_arriba == [0, 1, 0, 0, 0]:
        return "click"
    elif dedos_arriba == [0, 1, 1, 0, 0]:
        return "scroll_up"
    elif dedos_arriba == [0, 1, 1, 1, 1]:
        return "scroll_down"
    elif dedos_arriba == [1, 1, 1, 1, 1]:
        return "palma"
    elif dedos_arriba == [0, 0, 0, 0, 0]:
        return "puño"
    else:
        return "nada"

# Evitar acciones múltiples seguidas
ultimo_gesto = ""
tiempo_ultimo = time.time()

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Imagen espejo
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                gesto = detectar_gesto(hand_landmarks)

                # Mostrar el gesto
                cv2.putText(image, f"Gesto: {gesto}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Ejecutar acción si es nuevo gesto
                if gesto != "nada" and (gesto != ultimo_gesto or time.time() - tiempo_ultimo > 2):
                    if gesto == "click":
                        pyautogui.click()
                    elif gesto == "scroll_up":
                        pyautogui.scroll(300)
                    elif gesto == "scroll_down":
                        pyautogui.scroll(-300)
                    elif gesto == "puño":
                        pyautogui.hotkey('alt', 'tab')

                    ultimo_gesto = gesto
                    tiempo_ultimo = time.time()

        cv2.imshow('Control por gestos', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
