import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def dedos_arriba(hand_landmarks):
    dedos = []

    tips = [4, 8, 12, 16, 20]

    for i, tip in enumerate(tips):
        if i == 0:  # Pulgar
            if hand_landmarks.landmark[tip].x < hand_landmarks.landmark[tip - 1].x:
                dedos.append(1)
            else:
                dedos.append(0)
        else:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                dedos.append(1)
            else:
                dedos.append(0)
    return dedos

def detectar_gesto(dedos):
    if dedos == [1, 1, 0, 0, 0]:
        return "click_izquierdo"
    elif dedos == [0, 1, 1, 0, 0]:
        return "scroll_down"
    elif dedos == [0, 1, 1, 1, 0]:
        return "scroll_up"
    elif dedos == [0, 1, 1, 1, 1]:
        return "back"
    elif dedos == [1, 1, 1, 1, 1]:
        return "click_derecho"
    elif dedos == [0, 0, 0, 0, 0]:
        return "cerrar_pagina"
    elif dedos == [1, 0, 0, 0, 0]:
        return "restaurar_pagina"
    else:
        return "nada"

ultimo_gesto = ""
tiempo_ultimo = time.time()

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                dedos = dedos_arriba(hand_landmarks)
                gesto = detectar_gesto(dedos)

                cv2.putText(image, f"Gesto: {gesto}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if gesto != "nada" and (gesto != ultimo_gesto or time.time() - tiempo_ultimo > 1.5):
                    if gesto == "click_izquierdo":
                        pyautogui.click()
                    elif gesto == "click_derecho":
                        pyautogui.rightClick()
                    elif gesto == "scroll_down":
                        pyautogui.scroll(-300)
                    elif gesto == "scroll_up":
                        pyautogui.scroll(300)
                    elif gesto == "back":
                        pyautogui.hotkey('alt', 'left')
                    elif gesto == "cerrar_pagina":
                        pyautogui.hotkey('ctrl', 'w')
                    elif gesto == "restaurar_pagina":
                        pyautogui.hotkey('ctrl', 'shift', 't')

                    ultimo_gesto = gesto
                    tiempo_ultimo = time.time()

        cv2.imshow('Gesture Navigator 🖐️🧠', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
