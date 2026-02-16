import cv2
import mediapipe as mp
from utils import draw_button, detect_hover
from finance import run_finance, evaluate, reset
from medical import run_medical
from disaster import run_disaster

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

state = "MENU"

def get_hand_position(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)
            return (x, y)
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    hand_pos = get_hand_position(frame)

    if state == "MENU":

        # Holographic dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Title
        cv2.putText(frame, "LifeLens AR", (int(w*0.35), int(h*0.1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)

        btn1 = (int(w*0.1), int(h*0.25), int(w*0.3), int(h*0.32))
        btn2 = (int(w*0.1), int(h*0.4), int(w*0.3), int(h*0.47))
        btn3 = (int(w*0.1), int(h*0.55), int(w*0.3), int(h*0.62))

        draw_button(frame, btn1, "Financial")
        draw_button(frame, btn2, "Medical")
        draw_button(frame, btn3, "Disaster")

        if hand_pos:
            if detect_hover(hand_pos, btn1):
                state = "FINANCE"
            elif detect_hover(hand_pos, btn2):
                state = "MEDICAL"
            elif detect_hover(hand_pos, btn3):
                state = "DISASTER"

    elif state == "FINANCE":
        frame = run_finance(frame, hand_pos)

    elif state == "MEDICAL":
        state = run_medical(frame, hand_pos)

    elif state == "DISASTER":
        state = run_disaster(frame, hand_pos)

    cv2.imshow("LifeLens AR", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
