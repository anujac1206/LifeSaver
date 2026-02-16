"""
LifeLens AR - Main Application
Multi-world AR simulation with crash-safe architecture and professional UI
"""

import cv2
import mediapipe as mp
import sys
import os

# ================= SAFE WORLD IMPORT SYSTEM ================= #

def safe_import(module_name):
    """Import world module safely - won't crash if file doesn't exist"""
    try:
        module = __import__(module_name)
        run_func = getattr(module, f"run_{module_name}")
        eval_func = getattr(module, "evaluate", lambda: {})
        reset_func = getattr(module, "reset", lambda: None)
        print(f"✓ Loaded: {module_name}")
        return run_func, eval_func, reset_func, True
    except Exception as e:
        print(f"✗ {module_name} not found - continuing in safe mode")
        return (
            lambda frame, hand_pos: frame,
            lambda: {},
            lambda: None,
            False
        )

# Import all worlds
run_finance, eval_finance, reset_finance, finance_available = safe_import("finance_fixed")
run_medical, eval_medical, reset_medical, medical_available = safe_import("medical")
run_disaster, eval_disaster, reset_disaster, disaster_available = safe_import("disaster")
run_cybersecurity, eval_cyber, reset_cyber, cyber_available = safe_import("cybersecurity")
run_mental_health, eval_mental, reset_mental, mental_available = safe_import("mental_health")

# ================= WORLD REGISTRY ================= #

WORLD_REGISTRY = {
    "FINANCE": {
        "funcs": (run_finance, eval_finance, reset_finance),
        "available": finance_available,
        "display": "Financial Literacy",
        "description": "Master money management"
    },
    "MEDICAL": {
        "funcs": (run_medical, eval_medical, reset_medical),
        "available": medical_available,
        "display": "Medical Decisions",
        "description": "Navigate healthcare choices"
    },
    "DISASTER": {
        "funcs": (run_disaster, eval_disaster, reset_disaster),
        "available": disaster_available,
        "display": "Crisis Management",
        "description": "Survive emergencies"
    },
    "CYBERSECURITY": {
        "funcs": (run_cybersecurity, eval_cyber, reset_cyber),
        "available": cyber_available,
        "display": "Cyber Defense",
        "description": "Protect digital identity"
    },
    "MENTAL_HEALTH": {
        "funcs": (run_mental_health, eval_mental, reset_mental),
        "available": mental_available,
        "display": "Mental Wellness",
        "description": "Build resilience"
    }
}

# Global state
visited_worlds = set()
total_knowledge = 0
world_scores = {}

# Professional color palette (matching finance.py)
COLORS = {
    'slate_900': (15, 23, 42),
    'slate_800': (30, 41, 59),
    'slate_700': (51, 65, 85),
    'blue_500': (255, 171, 59),
    'blue_400': (251, 191, 96),
    'emerald_400': (178, 245, 129),
    'rose_400': (112, 118, 251),
    'white': (255, 255, 255),
    'gray_100': (243, 244, 246),
    'gray_300': (209, 213, 219),
}

# ================= MEDIAPIPE SETUP ================= #

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# State management
current_state = "MENU"
hover_start_time = 0
hover_duration = 1.5  # 1.5 seconds to select from menu

# ================= HAND TRACKING ================= #

def get_hand_position(frame):
    """Extract index finger position from webcam frame"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            # Index finger tip (landmark 8)
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            
            # Draw hand indicator
            cv2.circle(frame, (x, y), 12, COLORS['blue_500'], -1)
            cv2.circle(frame, (x, y), 16, COLORS['white'], 2)
            
            return (x, y)
    return None

# ================= MENU UI RENDERING ================= #

def render_menu(frame, hand_pos):
    """Render professional menu screen matching finance.py aesthetic"""
    h, w = frame.shape[:2]
    
    # Professional overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), COLORS['slate_900'], -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # ===== HEADER =====
    title = "LifeLens AR"
    title_scale = 2.2
    title_thickness = 4
    
    (title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thickness)
    title_x = (w - title_w) // 2
    title_y = 90
    
    # Title background panel
    panel_padding = 35
    cv2.rectangle(frame, (title_x - panel_padding, title_y - title_h - 20),
                 (title_x + title_w + panel_padding, title_y + 20),
                 COLORS['slate_800'], -1)
    cv2.rectangle(frame, (title_x - panel_padding, title_y - title_h - 20),
                 (title_x + title_w + panel_padding, title_y + 20),
                 COLORS['blue_500'], 3)
    
    # Title text
    cv2.putText(frame, title, (title_x, title_y),
               cv2.FONT_HERSHEY_SIMPLEX, title_scale, COLORS['white'], title_thickness, cv2.LINE_AA)
    
    # Accent line
    line_y = title_y + 30
    cv2.line(frame, (title_x, line_y), (title_x + title_w, line_y), COLORS['blue_500'], 3)
    
    # ===== STATS DISPLAY =====
    stats_y = 160
    
    # Worlds completed
    cv2.putText(frame, f"Worlds Completed: {len(visited_worlds)}/5", (50, stats_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS['gray_100'], 2, cv2.LINE_AA)
    
    # Total knowledge
    cv2.putText(frame, f"Total Knowledge: {total_knowledge}", (50, stats_y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS['emerald_400'], 2, cv2.LINE_AA)
    
    # ===== WORLD SELECTION CARDS =====
    available_worlds = [(key, data) for key, data in WORLD_REGISTRY.items() if data['available']]
    
    if not available_worlds:
        # No worlds available
        cv2.putText(frame, "No simulation worlds available", (w//2 - 250, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS['rose_400'], 2, cv2.LINE_AA)
        cv2.putText(frame, "Please ensure world modules are in the same directory", (w//2 - 350, h//2 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['gray_300'], 1, cv2.LINE_AA)
        return frame, None
    
    card_width = 420
    card_height = 100
    card_spacing = 25
    start_y = 250
    
    cards = []
    
    for i, (world_key, world_data) in enumerate(available_worlds):
        card_y = start_y + i * (card_height + card_spacing)
        card_x = (w - card_width) // 2
        
        is_hovered = check_hover(hand_pos, card_x, card_y, card_width, card_height)
        is_completed = world_key in visited_worlds
        
        render_world_card(frame, card_x, card_y, card_width, card_height, 
                         world_data['display'], world_data['description'],
                         is_hovered, is_completed, hand_pos)
        
        cards.append((card_x, card_y, card_width, card_height, world_key))
    
    # Instructions
    instructions_y = h - 60
    cv2.putText(frame, "Hover over a world for 1.5 seconds to begin", (w//2 - 280, instructions_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS['gray_300'], 1, cv2.LINE_AA)
    
    cv2.putText(frame, "Press ESC to quit", (w//2 - 100, instructions_y + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['gray_300'], 1, cv2.LINE_AA)
    
    return frame, cards

def render_world_card(frame, x, y, w, h, title, description, is_hovered, is_completed, hand_pos):
    """Render individual world selection card"""
    # Background
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS['slate_800'], -1)
    
    # Border
    border_color = COLORS['blue_500'] if is_hovered else COLORS['slate_700']
    border_thickness = 3 if is_hovered else 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, border_thickness, cv2.LINE_AA)
    
    # Completed badge
    if is_completed:
        badge_x = x + w - 40
        badge_y = y + 20
        cv2.circle(frame, (badge_x, badge_y), 12, COLORS['emerald_400'], -1)
        cv2.putText(frame, "✓", (badge_x - 7, badge_y + 6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 2)
    
    # Title
    cv2.putText(frame, title, (x + 25, y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLORS['white'], 2, cv2.LINE_AA)
    
    # Description
    cv2.putText(frame, description, (x + 25, y + 72),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['gray_300'], 1, cv2.LINE_AA)
    
    # Hover progress bar
    if is_hovered and hand_pos:
        import time
        global hover_start_time
        
        progress = min(1.0, (time.time() - hover_start_time) / hover_duration)
        
        bar_y = y + h - 8
        bar_width = w - 30
        
        # Track
        cv2.rectangle(frame, (x + 15, bar_y), (x + 15 + bar_width, bar_y + 4), 
                     COLORS['slate_700'], -1)
        
        # Fill
        fill = int(progress * bar_width)
        if fill > 0:
            cv2.rectangle(frame, (x + 15, bar_y), (x + 15 + fill, bar_y + 4), 
                         COLORS['blue_500'], -1)

def check_hover(hand_pos, x, y, w, h):
    """Check if hand is hovering over area"""
    if hand_pos is None:
        return False
    hx, hy = hand_pos
    return x <= hx <= x + w and y <= hy <= y + h

# ================= MAIN LOOP ================= #

print("\n" + "="*60)
print("LifeLens AR - Multi-World Simulation Platform")
print("="*60)
print("\nAvailable Worlds:")
for key, data in WORLD_REGISTRY.items():
    status = "✓" if data['available'] else "✗"
    print(f"  {status} {data['display']}")
print("\n" + "="*60 + "\n")

hovered_card = None
hover_confirmed = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera")
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Get hand position
    hand_pos = get_hand_position(frame)
    
    # ===== MENU STATE =====
    if current_state == "MENU":
        frame, cards = render_menu(frame, hand_pos)
        
        if cards:
            # Check for hover
            current_hover = None
            for card_x, card_y, card_w, card_h, world_key in cards:
                if check_hover(hand_pos, card_x, card_y, card_w, card_h):
                    current_hover = world_key
                    break
            
            # Update hover state
            if current_hover != hovered_card:
                hovered_card = current_hover
                import time
                hover_start_time = time.time()
                hover_confirmed = False
            
            # Check for selection
            if hovered_card and not hover_confirmed:
                import time
                if time.time() - hover_start_time >= hover_duration:
                    current_state = hovered_card
                    hover_confirmed = True
                    print(f"\n>>> Entering {WORLD_REGISTRY[hovered_card]['display']} <<<\n")
    
    # ===== WORLD STATE =====
    else:
        if current_state in WORLD_REGISTRY:
            world_data = WORLD_REGISTRY[current_state]
            
            if world_data['available']:
                run_func, eval_func, reset_func = world_data['funcs']
                
                # Run world simulation
                frame = run_func(frame, hand_pos)
                
                # Check for completion
                result = eval_func()
                if result and 'knowledge' in result and result['knowledge'] > 0:
                    # World completed
                    visited_worlds.add(current_state)
                    knowledge_gained = int(result['knowledge'])
                    total_knowledge += knowledge_gained
                    
                    world_scores[current_state] = result
                    
                    print(f"\n✓ Completed: {world_data['display']}")
                    print(f"  Knowledge Gained: +{knowledge_gained}")
                    print(f"  Total Knowledge: {total_knowledge}\n")
                    
                    # Reset world
                    reset_func()
                    
                    # Return to menu
                    current_state = "MENU"
                    hovered_card = None
            else:
                # World not available - return to menu
                current_state = "MENU"
        else:
            # Invalid state - return to menu
            current_state = "MENU"
    
    # Show frame
    cv2.imshow("LifeLens AR", frame)
    
    # Handle keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('m') or key == ord('M'):
        # Return to menu
        if current_state != "MENU":
            if current_state in WORLD_REGISTRY:
                _, _, reset_func = WORLD_REGISTRY[current_state]['funcs']
                reset_func()
        current_state = "MENU"
        hovered_card = None
        print("\n>>> Returned to Menu <<<\n")

# Cleanup
print("\n" + "="*60)
print("Session Summary:")
print("="*60)
print(f"Worlds Completed: {len(visited_worlds)}/5")
print(f"Total Knowledge: {total_knowledge}")
print("\nCompleted Worlds:")
for world in visited_worlds:
    print(f"  ✓ {WORLD_REGISTRY[world]['display']}")
print("="*60 + "\n")

cap.release()
cv2.destroyAllWindows()
hands.close()