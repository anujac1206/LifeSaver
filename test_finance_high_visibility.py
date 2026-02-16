"""
Test Script for HIGH VISIBILITY Finance Simulation
All UI elements are bright and clearly visible over webcam feed
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, '.')
from finance_high_visibility import run_finance, evaluate, reset

def create_mock_frame():
    """Create realistic webcam-like frame"""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Simulate darker webcam feed
    for i in range(720):
        color_val = int(30 + (i / 720) * 40)
        frame[i, :] = (color_val, color_val//2, color_val//3)
    
    # Add some noise for realism
    noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return frame

def main():
    """Run finance simulation test"""
    print("=" * 70)
    print("FINANCE SIMULATION - HIGH VISIBILITY VERSION")
    print("=" * 70)
    print("\n✨ KEY IMPROVEMENTS:")
    print("  • BRIGHT UI elements that stand out over webcam")
    print("  • HIGH CONTRAST borders and text")
    print("  • SOLID backgrounds (not transparent)")
    print("  • Light overlay (max 30% darkness)")
    print("  • Clear button visibility")
    print("\n🎮 CONTROLS:")
    print("  • Mouse: Hover over choices (hold 1 second to select)")
    print("  • R: Restart simulation")
    print("  • Q: Quit")
    print("=" * 70)
    print("\n🎯 TESTING CHECKLIST:")
    print("  [ ] Can you clearly see all three decision buttons?")
    print("  [ ] Is the text bright and readable?")
    print("  [ ] Are the borders visible (cyan/blue)?")
    print("  [ ] Does the progress bar show when hovering?")
    print("  [ ] Do stats update smoothly?")
    print("=" * 70)
    
    # Mouse state
    mouse_x, mouse_y = 0, 0
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_x, mouse_y = x, y
    
    cv2.namedWindow('Finance Simulation - High Visibility')
    cv2.setMouseCallback('Finance Simulation - High Visibility', mouse_callback)
    
    frame_count = 0
    
    print("\n🚀 Starting simulation...\n")
    
    while True:
        frame = create_mock_frame()
        hand_pos = (mouse_x, mouse_y)
        
        frame = run_finance(frame, hand_pos)
        
        # Add instruction overlay for first few seconds
        if frame_count < 150:
            cv2.rectangle(frame, (10, 670), (450, 710), (20, 30, 40), -1)
            cv2.rectangle(frame, (10, 670), (450, 710), (100, 200, 255), 2)
            cv2.putText(frame, "Hover over buttons for 1 second to select",
                       (20, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Finance Simulation - High Visibility', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n❌ Quitting...")
            break
        elif key == ord('r'):
            reset()
            print("\n🔄 Simulation restarted")
        
        frame_count += 1
    
    # Final evaluation
    results = evaluate()
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    for key, value in results.items():
        print(f"  {key:20s}: {value}")
    print("=" * 70)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()