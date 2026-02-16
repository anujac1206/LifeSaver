import cv2

from utils import draw_button, detect_hover

sequence = []
correct = ["Check", "Call", "CPR"]

def run_medical(frame, hand_pos):
    btn1 = (100, 200, 300, 260)
    btn2 = (100, 300, 300, 360)
    btn3 = (100, 400, 300, 460)

    draw_button(frame, btn1, "Check")
    draw_button(frame, btn2, "Call")
    draw_button(frame, btn3, "CPR")

    if hand_pos:
        if detect_hover(hand_pos, btn1):
            sequence.append("Check")
        elif detect_hover(hand_pos, btn2):
            sequence.append("Call")
        elif detect_hover(hand_pos, btn3):
            sequence.append("CPR")

    if len(sequence) == 3:
        sequence.clear()
        return "MENU"

    return "MEDICAL"
