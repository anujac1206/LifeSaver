import cv2
from utils import draw_button, detect_hover

risk = 30

def run_disaster(frame, hand_pos):
    global risk

    btn1 = (100, 200, 300, 260)
    btn2 = (100, 300, 300, 360)

    draw_button(frame, btn1, "Hide Under Table")
    draw_button(frame, btn2, "Stand Near Window")

    if hand_pos:
        if detect_hover(hand_pos, btn1):
            risk -= 10
            return reset()
        elif detect_hover(hand_pos, btn2):
            risk += 30
            return reset()

    return "DISASTER"

def reset():
    global risk
    risk = 30
    return "MENU"
