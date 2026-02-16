import cv2
import mediapipe as mp
from finance import run_finance

# ================= MEDIAPIPE SETUP ================= #

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= CAMERA SETUP ================= #

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ================= HAND TRACKING ================= #

def get_hand_position(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # draw pointer
            cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
            return (x, y)

    return None

# ================= MAIN LOOP ================= #

print("Running Finance Simulation Only")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    hand_pos = get_hand_position(frame)

    # Run finance world directly
    frame = run_finance(frame, hand_pos)

    cv2.imshow("Finance Simulation", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

# ================= CLEANUP ================= #

cap.release()
cv2.destroyAllWindows()
hands.close()
