import cv2
import time

hover_start = {}

def draw_button(frame, rect, text):
    x1, y1, x2, y2 = rect

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
    cv2.putText(frame, text, (x1+10,y1+40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

def detect_hover(hand_pos, rect, delay=1.0):
    x, y = hand_pos
    x1, y1, x2, y2 = rect

    if x1 < x < x2 and y1 < y < y2:
        if rect not in hover_start:
            hover_start[rect] = time.time()
        elif time.time() - hover_start[rect] > delay:
            hover_start.clear()
            return True
    else:
        if rect in hover_start:
            hover_start.pop(rect)

    return False
