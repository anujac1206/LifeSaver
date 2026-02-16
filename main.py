# import cv2
# import mediapipe as mp
# from utils import draw_button, detect_hover
# from finance import run_finance, evaluate, reset
# from medical import run_medical
# from disaster import run_disaster

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# state = "MENU"

# def get_hand_position(frame):
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)

#     if result.multi_hand_landmarks:
#         for handLms in result.multi_hand_landmarks:
#             h, w, _ = frame.shape
#             x = int(handLms.landmark[8].x * w)
#             y = int(handLms.landmark[8].y * h)
#             return (x, y)
#     return None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     h, w = frame.shape[:2]

#     hand_pos = get_hand_position(frame)

#     if state == "MENU":

#         # Holographic dark overlay
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
#         cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

#         # Title
#         cv2.putText(frame, "LifeLens AR", (int(w*0.35), int(h*0.1)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)

#         btn1 = (int(w*0.1), int(h*0.25), int(w*0.3), int(h*0.32))
#         btn2 = (int(w*0.1), int(h*0.4), int(w*0.3), int(h*0.47))
#         btn3 = (int(w*0.1), int(h*0.55), int(w*0.3), int(h*0.62))

#         draw_button(frame, btn1, "Financial")
#         draw_button(frame, btn2, "Medical")
#         draw_button(frame, btn3, "Disaster")

#         if hand_pos:
#             if detect_hover(hand_pos, btn1):
#                 state = "FINANCE"
#             elif detect_hover(hand_pos, btn2):
#                 state = "MEDICAL"
#             elif detect_hover(hand_pos, btn3):
#                 state = "DISASTER"

#     elif state == "FINANCE":
#         frame = run_finance(frame, hand_pos)

#     elif state == "MEDICAL":
#         state = run_medical(frame, hand_pos)

#     elif state == "DISASTER":
#         state = run_disaster(frame, hand_pos)

#     cv2.imshow("LifeLens AR", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp

# # ---------------- SAFE IMPORTS ----------------

# def safe_import(module_name, function_name):
#     try:
#         module = __import__(module_name)
#         return getattr(module, function_name)
#     except Exception:
#         print(f"[WARNING] {module_name}.py not found. Using fallback.")
#         return None

# # Try importing all modules safely
# run_finance = safe_import("finance", "run_finance")
# run_medical = safe_import("medical", "run_medical")
# run_disaster = safe_import("disaster", "run_disaster")
# run_cyber = safe_import("cybersecurity", "run_cyber")
# run_environment = safe_import("environment", "run_environment")
# run_mental = safe_import("mental", "run_mental")
# run_productivity = safe_import("productivity", "run_productivity")
# run_emergency = safe_import("emergency", "run_emergency")

# # ---------------- MEDIAPIPE ----------------

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# state = "MENU"

# # ---------------- HAND TRACKING ----------------

# def get_hand_position(frame):
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)

#     if result.multi_hand_landmarks:
#         for handLms in result.multi_hand_landmarks:
#             h, w, _ = frame.shape
#             x = int(handLms.landmark[8].x * w)
#             y = int(handLms.landmark[8].y * h)
#             return (x, y)
#     return None

# # ---------------- BUTTON UTILS ----------------

# def draw_button(frame, coords, text):
#     x1, y1, x2, y2 = coords
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
#     cv2.putText(frame, text, (x1 + 10, y1 + 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

# def detect_hover(hand_pos, btn):
#     x1, y1, x2, y2 = btn
#     return x1 < hand_pos[0] < x2 and y1 < hand_pos[1] < y2

# # ---------------- FALLBACK SCREEN ----------------

# def module_not_available(frame, module_name):
#     h, w = frame.shape[:2]

#     cv2.putText(frame, f"{module_name} Module Not Available",
#                 (int(w*0.2), int(h*0.3)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1,
#                 (0,0,255), 3)

#     cv2.putText(frame, "File Missing or Not Created",
#                 (int(w*0.3), int(h*0.4)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1,
#                 (255,255,255), 2)

#     return module_name

# # ---------------- MAIN LOOP ----------------

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     h, w = frame.shape[:2]
#     hand_pos = get_hand_position(frame)

#     if state == "MENU":

#         overlay = frame.copy()
#         cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
#         cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

#         cv2.putText(frame, "LifeLens AR",
#                     (int(w*0.35), int(h*0.08)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5,
#                     (0,255,255), 3)

#         modules = [
#             "FINANCE",
#             "MEDICAL",
#             "DISASTER",
#             "CYBER",
#             "ENVIRONMENT",
#             "MENTAL",
#             "PRODUCTIVITY",
#             "EMERGENCY"
#         ]

#         buttons = []

#         for i, module in enumerate(modules):
#             y1 = int(h*(0.18 + i*0.08))
#             y2 = int(h*(0.24 + i*0.08))
#             btn = (int(w*0.1), y1, int(w*0.35), y2)
#             buttons.append((btn, module))
#             draw_button(frame, btn, module.title())

#         if hand_pos:
#             for btn, module in buttons:
#                 if detect_hover(hand_pos, btn):
#                     state = module

#     # ---------------- MODULE HANDLING ----------------

#     elif state == "FINANCE":
#         if run_finance:
#             frame = run_finance(frame, hand_pos)
#         else:
#             frame = module_not_available(frame, "FINANCE")

#     elif state == "MEDICAL":
#         if run_medical:
#             state = run_medical(frame, hand_pos)
#         else:
#             frame = module_not_available(frame, "MEDICAL")

#     elif state == "DISASTER":
#         if run_disaster:
#             state = run_disaster(frame, hand_pos)
#         else:
#             frame = module_not_available(frame, "DISASTER")

#     elif state == "CYBER":
#         if run_cyber:
#             state = run_cyber(frame, hand_pos)
#         else:
#             frame = module_not_available(frame, "CYBER")

#     elif state == "ENVIRONMENT":
#         if run_environment:
#             state = run_environment(frame, hand_pos)
#         else:
#             frame = module_not_available(frame, "ENVIRONMENT")

#     elif state == "MENTAL":
#         if run_mental:
#             state = run_mental(frame, hand_pos)
#         else:
#             frame = module_not_available(frame, "MENTAL")

#     elif state == "PRODUCTIVITY":
#         if run_productivity:
#             state = run_productivity(frame, hand_pos)
#         else:
#             frame = module_not_available(frame, "PRODUCTIVITY")

#     elif state == "EMERGENCY":
#         if run_emergency:
#             state = run_emergency(frame, hand_pos)
#         else:
#             frame = module_not_available(frame, "EMERGENCY")

#     cv2.imshow("LifeLens AR", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import time
import importlib.util
import sys

# ==================== COLOR PALETTE ====================
SLATE_900 = (15, 23, 42)
SLATE_800 = (30, 41, 59)
SLATE_700 = (51, 65, 85)
ACCENT_BLUE = (59, 171, 255)
ACCENT_GREEN = (129, 245, 178)
ACCENT_PURPLE = (112, 118, 250)
WHITE = (255, 255, 255)
GRAY_300 = (219, 213, 209)

# ==================== STATE MANAGEMENT ====================
STATE = "MENU"
current_module = None
hover_start = {}
pulse_time = 0

# Module definitions
MODULES = [
    {"id": "FINANCE", "file": "finance", "title": "Finance", "subtitle": "Portfolio Management"},
    {"id": "MEDICAL", "file": "medical", "title": "Medical", "subtitle": "Health Diagnostics"},
    {"id": "DISASTER", "file": "disaster", "title": "Disaster", "subtitle": "Emergency Response"},
    {"id": "CYBER", "file": "cybersecurity", "title": "Cybersecurity", "subtitle": "Threat Detection"},
    {"id": "ENVIRONMENT", "file": "environment", "title": "Environment", "subtitle": "Climate Monitoring"},
    {"id": "MENTAL", "file": "mental_health", "title": "Mental Health", "subtitle": "Wellness Tracker"},
    {"id": "PRODUCTIVITY", "file": "productivity", "title": "Productivity", "subtitle": "Task Optimizer"},
    {"id": "EMERGENCY", "file": "emergency", "title": "Emergency", "subtitle": "Crisis Management"}
]

# ==================== UTILITY FUNCTIONS ====================
def safe_import(module_name):
    """Safely import a module by name"""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return None
        module = importlib.import_module(module_name)
        return module
    except Exception:
        return None

def get_hand_position(hand_landmarks, frame_width, frame_height):
    """Get normalized hand position from landmarks"""
    if hand_landmarks:
        index_finger = hand_landmarks.landmark[8]
        x = int(index_finger.x * frame_width)
        y = int(index_finger.y * frame_height)
        return (x, y)
    return None

def blend_alpha(bg_color, fg_color, alpha):
    """Blend two colors with alpha"""
    return tuple(int(bg * (1 - alpha) + fg * alpha) for bg, fg in zip(bg_color, fg_color))

# ==================== RENDERING FUNCTIONS ====================
def render_background(frame):
    """Render animated gradient background"""
    global pulse_time
    h, w = frame.shape[:2]
    
    # Dark base
    frame[:] = SLATE_900
    
    # Animated gradient overlay
    pulse_time += 0.02
    for y in range(h):
        intensity = (np.sin(pulse_time + y * 0.003) + 1) * 0.5
        color = blend_alpha(SLATE_900, SLATE_800, intensity * 0.3)
        cv2.line(frame, (0, y), (w, y), color, 1)

def render_glass_panel(frame, x, y, w, h, alpha=0.15, border_color=GRAY_300, border_width=1):
    """Render a glassmorphism panel"""
    overlay = frame.copy()
    
    # Glass background
    cv2.rectangle(overlay, (x, y), (x + w, y + h), SLATE_800, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, border_width)

def render_title_panel(frame):
    """Render centered title section with pulse animation"""
    global pulse_time
    h, w = frame.shape[:2]
    
    panel_w = 600
    panel_h = 120
    panel_x = (w - panel_w) // 2
    panel_y = 40
    
    # Pulse animation
    pulse = (np.sin(pulse_time * 2) + 1) * 0.5
    border_color = blend_alpha(ACCENT_BLUE, WHITE, pulse * 0.3)
    
    # Glass panel
    render_glass_panel(frame, panel_x, panel_y, panel_w, panel_h, alpha=0.2, 
                      border_color=border_color, border_width=2)
    
    # Title
    title = "LifeLens AR"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)[0]
    title_x = (w - title_size[0]) // 2
    title_y = panel_y + 50
    
    # Title shadow
    cv2.putText(frame, title, (title_x + 2, title_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 
                1.8, SLATE_900, 3, cv2.LINE_AA)
    # Title main
    cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, 
                1.8, WHITE, 3, cv2.LINE_AA)
    
    # Subtitle
    subtitle = "Gesture Controlled Decision Simulator"
    sub_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    sub_x = (w - sub_size[0]) // 2
    sub_y = panel_y + 90
    cv2.putText(frame, subtitle, (sub_x, sub_y), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, GRAY_300, 1, cv2.LINE_AA)

def render_module_card(frame, module, x, y, w, h, is_hovered, hover_progress):
    """Render a single module card with hover effects"""
    
    # Base card with glass effect
    alpha = 0.25 if is_hovered else 0.15
    border_color = ACCENT_BLUE if is_hovered else GRAY_300
    border_width = 2 if is_hovered else 1
    
    # Brightness boost on hover
    overlay = frame.copy()
    card_color = blend_alpha(SLATE_800, SLATE_700, 0.5 if is_hovered else 0)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), card_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, border_width)
    
    # Icon placeholder circle
    icon_radius = 30
    icon_x = x + w // 2
    icon_y = y + 60
    icon_color = blend_alpha(SLATE_700, ACCENT_BLUE, 0.3 if is_hovered else 0.1)
    cv2.circle(frame, (icon_x, icon_y), icon_radius, icon_color, -1)
    cv2.circle(frame, (icon_x, icon_y), icon_radius, border_color, 2)
    
    # Title
    title = module["title"]
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    title_x = x + (w - title_size[0]) // 2
    title_y = y + 130
    cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, WHITE, 2, cv2.LINE_AA)
    
    # Subtitle
    subtitle = module["subtitle"]
    sub_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    sub_x = x + (w - sub_size[0]) // 2
    sub_y = y + 160
    cv2.putText(frame, subtitle, (sub_x, sub_y), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, GRAY_300, 1, cv2.LINE_AA)
    
    # Progress bar on hover
    if is_hovered and hover_progress > 0:
        bar_y = y + h - 8
        bar_height = 4
        bar_margin = 10
        bar_width = int((w - 2 * bar_margin) * hover_progress)
        
        # Progress bar background
        cv2.rectangle(frame, (x + bar_margin, bar_y), 
                     (x + w - bar_margin, bar_y + bar_height), SLATE_700, -1)
        
        # Progress bar fill with gradient
        progress_color = blend_alpha(ACCENT_BLUE, ACCENT_GREEN, hover_progress)
        cv2.rectangle(frame, (x + bar_margin, bar_y), 
                     (x + bar_margin + bar_width, bar_y + bar_height), 
                     progress_color, -1)

def render_menu(frame, hand_pos):
    """Render the main menu with module cards"""
    global hover_start, STATE, current_module
    
    h, w = frame.shape[:2]
    
    # Grid layout
    cols = 2
    rows = 4
    card_w = 280
    card_h = 200
    gap_x = 40
    gap_y = 30
    
    grid_w = cols * card_w + (cols - 1) * gap_x
    grid_h = rows * card_h + (rows - 1) * gap_y
    start_x = (w - grid_w) // 2
    start_y = 200
    
    current_time = time.time()
    
    for idx, module in enumerate(MODULES):
        row = idx // cols
        col = idx % cols
        
        card_x = start_x + col * (card_w + gap_x)
        card_y = start_y + row * (card_h + gap_y)
        
        # Check hover
        is_hovered = False
        hover_progress = 0
        
        if hand_pos:
            hx, hy = hand_pos
            if card_x <= hx <= card_x + card_w and card_y <= hy <= card_y + card_h:
                is_hovered = True
                
                # Start or continue hover timer
                if module["id"] not in hover_start:
                    hover_start[module["id"]] = current_time
                
                elapsed = current_time - hover_start[module["id"]]
                hover_progress = min(elapsed / 3.0, 1.0)
                
                # Trigger after 3 seconds
                if hover_progress >= 1.0:
                    # Check if module exists
                    imported_module = safe_import(module["file"])
                    if imported_module:
                        STATE = module["id"]
                        current_module = imported_module
                    else:
                        STATE = "MISSING"
                        current_module = module
                    hover_start.clear()
            else:
                # Reset hover for this module
                if module["id"] in hover_start:
                    del hover_start[module["id"]]
        else:
            # No hand detected, reset all hovers
            if module["id"] in hover_start:
                del hover_start[module["id"]]
        
        render_module_card(frame, module, card_x, card_y, card_w, card_h, 
                          is_hovered, hover_progress)

def render_missing_overlay(frame, module):
    """Render overlay when module is not installed"""
    h, w = frame.shape[:2]
    
    # Darken background
    overlay = frame.copy()
    overlay[:] = blend_alpha(overlay[:], SLATE_900, 0.7)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Center panel
    panel_w = 500
    panel_h = 250
    panel_x = (w - panel_w) // 2
    panel_y = (h - panel_h) // 2
    
    render_glass_panel(frame, panel_x, panel_y, panel_w, panel_h, 
                      alpha=0.3, border_color=ACCENT_PURPLE, border_width=2)
    
    # Icon
    icon_x = panel_x + panel_w // 2
    icon_y = panel_y + 70
    cv2.circle(frame, (icon_x, icon_y), 35, ACCENT_PURPLE, -1)
    cv2.putText(frame, "!", (icon_x - 12, icon_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, WHITE, 3, cv2.LINE_AA)
    
    # Title
    title = "Module Not Installed"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    title_x = panel_x + (panel_w - title_size[0]) // 2
    cv2.putText(frame, title, (title_x, panel_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, WHITE, 2, cv2.LINE_AA)
    
    # Message
    msg = f"{module['title']} module is not available"
    msg_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    msg_x = panel_x + (panel_w - msg_size[0]) // 2
    cv2.putText(frame, msg, (msg_x, panel_y + 175), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, GRAY_300, 1, cv2.LINE_AA)
    
    # Back instruction
    back = "Move hand away to return"
    back_size = cv2.getTextSize(back, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    back_x = panel_x + (panel_w - back_size[0]) // 2
    cv2.putText(frame, back, (back_x, panel_y + 210), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, ACCENT_BLUE, 1, cv2.LINE_AA)

# ==================== MAIN LOOP ====================
def main():
    global STATE, current_module
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Process hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        hand_pos = None
        if results.multi_hand_landmarks:
            hand_pos = get_hand_position(results.multi_hand_landmarks[0], w, h)
        
        # Render background
        render_background(frame)
        
        # State machine
        if STATE == "MENU":
            render_title_panel(frame)
            render_menu(frame, hand_pos)
            
        elif STATE == "MISSING":
            render_title_panel(frame)
            render_missing_overlay(frame, current_module)
            
            # Return to menu when hand leaves
            if hand_pos is None:
                STATE = "MENU"
                current_module = None
                
        else:
            # Module is loaded
            if current_module and hasattr(current_module, 'run'):
                try:
                    result = current_module.run(frame, hand_pos)
                    if result == "MENU":
                        STATE = "MENU"
                        current_module = None
                except Exception as e:
                    # Handle module errors gracefully
                    STATE = "MENU"
                    current_module = None
        
        # Display
        cv2.imshow('LifeLens AR', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()