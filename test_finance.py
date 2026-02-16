"""
Test Script for Finance Simulation Module
Demonstrates all phases with keyboard controls
"""

import cv2
import numpy as np
from finance import run_finance, evaluate, reset

def create_mock_frame():
    """Create a mock webcam frame"""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Add gradient background
    for i in range(720):
        color_val = int(20 + (i / 720) * 30)
        frame[i, :] = (color_val, color_val//2, color_val//3)
    return frame

def main():
    """Run finance simulation test"""
    print("=" * 60)
    print("FINANCE SIMULATION - TEST MODE")
    print("=" * 60)
    print("\nControls:")
    print("  Mouse: Hover over buttons to select")
    print("  1-4: Quick select button 1-4")
    print("  R: Restart simulation")
    print("  Q: Quit")
    print("=" * 60)
    
    # Mouse state
    mouse_x, mouse_y = 0, 0
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_x, mouse_y = x, y
    
    # Create window
    cv2.namedWindow('Finance Simulation')
    cv2.setMouseCallback('Finance Simulation', mouse_callback)
    
    frame_count = 0
    
    while True:
        # Create frame
        frame = create_mock_frame()
        
        # Get hand position from mouse
        hand_pos = (mouse_x, mouse_y)
        
        # Run simulation
        frame = run_finance(frame, hand_pos)
        
        # Add instruction overlay
        if frame_count < 180:  # Show for 3 seconds
            cv2.putText(frame, "Hover over choices for 1 second to select", 
                       (20, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 180), 2)
        
        # Show frame
        cv2.imshow('Finance Simulation', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset()
            print("\n[RESET] Simulation restarted")
        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            # Quick selection for testing
            pass
        
        frame_count += 1
    
    # Final evaluation
    results = evaluate()
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    for key, value in results.items():
        print(f"{key:20s}: {value}")
    print("=" * 60)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()