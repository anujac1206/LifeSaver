"""
MEDICAL DECISION SIMULATION - MAIN ENTRY POINT
Professional high-visibility interface with hand tracking
Supports both webcam hand tracking and mouse fallback
"""

"""
MEDICAL DECISION SIMULATION - MAIN ENTRY POINT (OPTIMIZED)
Fast-loading professional interface with improved hand tracking
"""


"""
MEDICAL DECISION SIMULATION - MAIN ENTRY POINT
Professional high-visibility interface with hand tracking
Supports both webcam hand tracking and mouse fallback
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import sys

# Import the medical simulation module
from medical_with_images import (
    MedicalDecisionSimulation,
    run_medical,
    COLORS
)

# ============================================================================
# HAND TRACKING (MediaPipe)
# ============================================================================
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available. Install with: pip install mediapipe")
    print("Falling back to mouse control...")

class HandTracker:
    """Hand tracking using MediaPipe"""
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            self.hands = None
            self.enabled = False
            return
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.enabled = True
    
    def process(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Process frame and return index finger tip position
        Returns (x, y) or None if no hand detected
        """
        if not self.enabled:
            return None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Index finger tip is landmark 8
            index_tip = hand_landmarks.landmark[8]
            
            h, w = frame.shape[:2]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)
            
            return (x, y)
        
        return None
    
    def release(self):
        """Release MediaPipe resources"""
        if self.enabled and self.hands:
            self.hands.close()

# ============================================================================
# MOUSE FALLBACK SYSTEM
# ============================================================================
class MouseTracker:
    """Mouse-based control as fallback for hand tracking"""
    def __init__(self):
        self.position: Optional[Tuple[int, int]] = None
        self.window_name = "Medical Decision Simulation"
    
    def mouse_callback(self, event, x, y, flags, param):
        """OpenCV mouse callback"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.position = (x, y)
    
    def setup(self, window_name: str):
        """Setup mouse callback for window"""
        self.window_name = window_name
        cv2.setMouseCallback(window_name, self.mouse_callback)
    
    def get_position(self) -> Optional[Tuple[int, int]]:
        """Get current mouse position"""
        return self.position

# ============================================================================
# APPLICATION CLASS
# ============================================================================
class MedicalSimulationApp:
    """Main application controller"""
    def __init__(self, use_webcam: bool = True, camera_id: int = 0):
        self.use_webcam = use_webcam
        self.camera_id = camera_id
        
        # Initialize video capture
        if use_webcam:
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {camera_id}")
                print("Falling back to demo mode...")
                self.use_webcam = False
        
        # Initialize tracking systems
        self.hand_tracker = HandTracker() if MEDIAPIPE_AVAILABLE else None
        self.mouse_tracker = MouseTracker()
        
        # Initialize simulation
        self.simulation = MedicalDecisionSimulation()
        
        # Window setup
        self.window_name = "Medical Decision Simulation - Professional Edition"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        # Setup mouse callback
        self.mouse_tracker.setup(self.window_name)
        
        # Control state
        self.running = True
        self.paused = False
        self.show_help = False
    
    def get_frame(self) -> np.ndarray:
        """Get current video frame or generate demo frame"""
        if self.use_webcam:
            ret, frame = self.cap.read()
            if not ret:
                # Generate blank frame if camera fails
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                frame[:] = (60, 60, 70)  # Dark gray
                cv2.putText(frame, "CAMERA ERROR", (500, 360),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            return frame
        else:
            # Demo mode - generate gradient background
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            for i in range(720):
                intensity = int(40 + (i / 720) * 30)
                frame[i, :] = (intensity, intensity, intensity + 10)
            
            # Add demo text
            cv2.putText(frame, "DEMO MODE - Use Mouse for Control", (380, 360),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            return frame
    
    def get_cursor_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Get cursor position from hand tracking or mouse"""
        # Try hand tracking first
        if self.hand_tracker and self.hand_tracker.enabled:
            hand_pos = self.hand_tracker.process(frame)
            if hand_pos:
                return hand_pos
        
        # Fall back to mouse
        return self.mouse_tracker.get_position()
    
    def draw_help_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw help overlay with controls"""
        overlay = frame.copy()
        
        # Semi-transparent background
        cv2.rectangle(overlay, (340, 150), (940, 550), COLORS['bg_primary'], -1)
        cv2.rectangle(overlay, (340, 150), (940, 550), COLORS['accent_blue'], 3)
        
        # Title
        cv2.putText(overlay, "CONTROLS & HELP", (450, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS['text_primary'], 2, cv2.LINE_AA)
        
        # Instructions
        instructions = [
            "NAVIGATION:",
            "  - Move hand or mouse over decisions to preview",
            "  - Hold for 3 seconds to confirm selection",
            "",
            "KEYBOARD CONTROLS:",
            "  H - Toggle this help screen",
            "  P - Pause/Resume simulation",
            "  R - Restart simulation",
            "  Q/ESC - Quit application",
            "",
            "GOAL:",
            "  Balance patient care, infection control,",
            "  resources, staff morale, and public trust",
            "  through strategic hospital management decisions.",
        ]
        
        y_offset = 250
        for line in instructions:
            if line.startswith("  "):
                # Indented line
                cv2.putText(overlay, line, (380, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_secondary'], 1, cv2.LINE_AA)
            elif line == "":
                pass  # Skip empty lines
            else:
                # Header line
                cv2.putText(overlay, line, (370, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['accent_teal'], 1, cv2.LINE_AA)
            y_offset += 25
        
        # Close instruction
        cv2.putText(overlay, "Press H to close", (540, 530),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_dim'], 1, cv2.LINE_AA)
        
        return cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
    
    def draw_status_bar(self, frame: np.ndarray):
        """Draw status bar at the top"""
        # Status text
        status_text = "HAND TRACKING" if (self.hand_tracker and self.hand_tracker.enabled) else "MOUSE CONTROL"
        cv2.putText(frame, status_text, (frame.shape[1] - 200, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['accent_green'], 1, cv2.LINE_AA)
        
        if self.paused:
            cv2.putText(frame, "PAUSED", (frame.shape[1] - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['accent_red'], 1, cv2.LINE_AA)
        
        # Help hint
        cv2.putText(frame, "Press H for help", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text_dim'], 1, cv2.LINE_AA)
    
    def handle_keyboard(self, key: int) -> bool:
        """
        Handle keyboard input
        Returns False if should quit
        """
        if key == ord('q') or key == 27:  # Q or ESC
            return False
        elif key == ord('h') or key == ord('H'):
            self.show_help = not self.show_help
        elif key == ord('p') or key == ord('P'):
            self.paused = not self.paused
        elif key == ord('r') or key == ord('R'):
            self.simulation.reset()
            print("Simulation reset")
        
        return True
    
    def run(self):
        """Main application loop"""
        print("=" * 60)
        print("MEDICAL DECISION SIMULATION - PROFESSIONAL EDITION")
        print("=" * 60)
        print(f"Tracking mode: {'Hand Tracking' if (self.hand_tracker and self.hand_tracker.enabled) else 'Mouse Control'}")
        print(f"Video source: {'Webcam' if self.use_webcam else 'Demo Mode'}")
        print("")
        print("Controls:")
        print("  H - Help")
        print("  P - Pause/Resume")
        print("  R - Restart")
        print("  Q/ESC - Quit")
        print("=" * 60)
        
        try:
            while self.running:
                # Get video frame
                frame = self.get_frame()
                
                # Get cursor position
                cursor_pos = self.get_cursor_position(frame)
                
                # Render simulation (unless paused)
                if not self.paused:
                    display_frame = run_medical(frame, cursor_pos, self.simulation)
                else:
                    display_frame = frame.copy()
                
                # Add status bar
                self.draw_status_bar(display_frame)
                
                # Show help overlay if enabled
                if self.show_help:
                    display_frame = self.draw_help_overlay(display_frame)
                
                # Display frame
                cv2.imshow(self.window_name, display_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key pressed
                    if not self.handle_keyboard(key):
                        break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        
        if self.use_webcam and self.cap:
            self.cap.release()
        
        if self.hand_tracker:
            self.hand_tracker.release()
        
        cv2.destroyAllWindows()
        print("Shutdown complete")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def print_usage():
    """Print usage information"""
    print("Medical Decision Simulation - Professional Edition")
    print("")
    print("Usage:")
    print("  python main.py [options]")
    print("")
    print("Options:")
    print("  --demo          Run in demo mode (no webcam)")
    print("  --camera ID     Use specific camera ID (default: 0)")
    print("  --help          Show this help message")
    print("")
    print("Controls:")
    print("  H               Toggle help overlay")
    print("  P               Pause/Resume")
    print("  R               Restart simulation")
    print("  Q/ESC           Quit")
    print("")
    print("Requirements:")
    print("  - OpenCV (cv2)")
    print("  - NumPy")
    print("  - MediaPipe (optional, for hand tracking)")
    print("")
    print("Install dependencies:")
    print("  pip install opencv-python numpy mediapipe")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Main entry point"""
    # Parse command line arguments
    use_webcam = True
    camera_id = 0
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--help":
            print_usage()
            return
        elif arg == "--demo":
            use_webcam = False
        elif arg == "--camera":
            if i + 1 < len(sys.argv) - 1:
                try:
                    camera_id = int(sys.argv[i + 2])
                except ValueError:
                    print(f"Error: Invalid camera ID '{sys.argv[i + 2]}'")
                    return
    
    # Create and run application
    app = MedicalSimulationApp(use_webcam=use_webcam, camera_id=camera_id)
    app.run()

if __name__ == "__main__":
    main()