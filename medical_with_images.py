"""
MEDICAL DECISION SIMULATION - PROFESSIONAL HIGH VISIBILITY EDITION
Advanced hospital administration and crisis management system with image support
Designed for maximum webcam visibility and professional presentation
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import os

# ============================================================================
# COLOR PALETTE - Professional Medical Theme
# ============================================================================
COLORS = {
    'bg_primary': (45, 45, 50),           # Dark slate background
    'bg_secondary': (55, 55, 65),         # Lighter slate
    'bg_panel': (35, 35, 40),             # Panel background
    'accent_blue': (180, 120, 60),        # Medical blue accent
    'accent_green': (100, 180, 80),       # Health green
    'accent_red': (80, 80, 200),          # Alert red
    'accent_teal': (200, 180, 80),        # Teal accent
    'text_primary': (240, 240, 245),      # Light text
    'text_secondary': (180, 185, 200),    # Dimmed text
    'text_dim': (140, 145, 160),          # Very dim text
    'hover_overlay': (90, 120, 150),      # Hover highlight
    'stat_critical': (70, 70, 220),       # Critical stat (red)
    'stat_warning': (80, 180, 220),       # Warning stat (orange)
    'stat_good': (90, 200, 120),          # Good stat (green)
    'stat_excellent': (120, 220, 100),    # Excellent stat (bright green)
}

# ============================================================================
# IMAGE MANAGEMENT SYSTEM
# ============================================================================
class ImageManager:
    """Manages loading and caching of scenario images"""
    def __init__(self, images_folder: str = "medical_images"):
        self.images_folder = images_folder
        self.image_cache = {}
        self.placeholder_cache = {}
        
        # Create images folder if it doesn't exist
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
            print(f"📁 Created images folder: {images_folder}")
            print(f"   Please add PNG images for each scenario:")
            print(f"   - emergency_room.png")
            print(f"   - infection_control.png")
            print(f"   - equipment_allocation.png")
            print(f"   - crisis_event.png")
            print(f"   - long_term_policy.png")
            print(f"   - mass_casualty.png")
            print(f"   - outbreak.png")
            print(f"   - power_failure.png")
    
    def load_image(self, image_key: str, width: int, height: int) -> np.ndarray:
        """
        Load and resize image from disk, with caching
        Returns the image or a placeholder if not found
        """
        cache_key = f"{image_key}_{width}_{height}"
        
        # Check cache first
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]
        
        # Try to load from disk
        image_path = os.path.join(self.images_folder, f"{image_key}.png")
        
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                # Resize to target dimensions
                img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                self.image_cache[cache_key] = img_resized
                return img_resized
        
        # Return placeholder if image not found
        return self._create_placeholder(image_key, width, height)
    
    def _create_placeholder(self, image_key: str, width: int, height: int) -> np.ndarray:
        """Create a placeholder image with icon and text"""
        cache_key = f"placeholder_{image_key}_{width}_{height}"
        
        if cache_key in self.placeholder_cache:
            return self.placeholder_cache[cache_key]
        
        # Create gradient background
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create medical-themed gradient
        for i in range(height):
            intensity_blue = int(40 + (i / height) * 30)
            intensity_green = int(50 + (i / height) * 20)
            placeholder[i, :] = (intensity_blue + 20, intensity_green + 10, intensity_blue)
        
        # Add icon based on image key
        self._add_icon(placeholder, image_key, width, height)
        
        # Add text label
        label = image_key.replace('_', ' ').upper()
        
        # Wrap text if too long
        if len(label) > 20:
            words = label.split()
            line1 = ' '.join(words[:len(words)//2])
            line2 = ' '.join(words[len(words)//2:])
            
            cv2.putText(placeholder, line1, (width//2 - 80, height//2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 220), 2, cv2.LINE_AA)
            cv2.putText(placeholder, line2, (width//2 - 80, height//2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 220), 2, cv2.LINE_AA)
        else:
            cv2.putText(placeholder, label, (width//2 - 100, height//2 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 220), 2, cv2.LINE_AA)
        
        # Add "Add image" hint
        cv2.putText(placeholder, f"Add: {image_key}.png", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 160), 1, cv2.LINE_AA)
        
        self.placeholder_cache[cache_key] = placeholder
        return placeholder
    
    def _add_icon(self, img: np.ndarray, image_key: str, width: int, height: int):
        """Add themed icon to placeholder"""
        center_x = width // 2
        center_y = height // 2 - 40
        
        if 'emergency' in image_key or 'casualty' in image_key:
            # Red cross symbol
            cv2.rectangle(img, (center_x - 5, center_y - 25), (center_x + 5, center_y + 25), 
                         (80, 80, 220), -1)
            cv2.rectangle(img, (center_x - 25, center_y - 5), (center_x + 25, center_y + 5), 
                         (80, 80, 220), -1)
        
        elif 'infection' in image_key or 'outbreak' in image_key:
            # Biohazard-style circles
            cv2.circle(img, (center_x, center_y), 25, (100, 180, 220), 3)
            cv2.circle(img, (center_x, center_y), 15, (100, 180, 220), 3)
            cv2.circle(img, (center_x, center_y), 5, (100, 180, 220), -1)
        
        elif 'equipment' in image_key or 'allocation' in image_key:
            # Equipment boxes
            cv2.rectangle(img, (center_x - 30, center_y - 20), (center_x - 10, center_y + 20), 
                         (180, 160, 100), 2)
            cv2.rectangle(img, (center_x - 5, center_y - 20), (center_x + 15, center_y + 20), 
                         (180, 160, 100), 2)
            cv2.rectangle(img, (center_x + 20, center_y - 20), (center_x + 40, center_y + 20), 
                         (180, 160, 100), 2)
        
        elif 'power' in image_key or 'failure' in image_key:
            # Lightning bolt
            points = np.array([
                [center_x, center_y - 30],
                [center_x - 10, center_y],
                [center_x + 5, center_y],
                [center_x - 5, center_y + 30],
                [center_x + 15, center_y + 5],
                [center_x, center_y + 5]
            ], np.int32)
            cv2.fillPoly(img, [points], (100, 200, 220))
        
        elif 'policy' in image_key or 'long_term' in image_key:
            # Document icon
            cv2.rectangle(img, (center_x - 20, center_y - 25), (center_x + 20, center_y + 25), 
                         (180, 200, 180), 2)
            cv2.line(img, (center_x - 15, center_y - 15), (center_x + 15, center_y - 15), 
                    (180, 200, 180), 2)
            cv2.line(img, (center_x - 15, center_y - 5), (center_x + 15, center_y - 5), 
                    (180, 200, 180), 2)
            cv2.line(img, (center_x - 15, center_y + 5), (center_x + 15, center_y + 5), 
                    (180, 200, 180), 2)
            cv2.line(img, (center_x - 15, center_y + 15), (center_x + 15, center_y + 15), 
                    (180, 200, 180), 2)
        
        else:
            # Generic medical cross
            cv2.circle(img, (center_x, center_y), 30, (120, 180, 160), 3)
            cv2.rectangle(img, (center_x - 3, center_y - 18), (center_x + 3, center_y + 18), 
                         (120, 180, 160), -1)
            cv2.rectangle(img, (center_x - 18, center_y - 3), (center_x + 18, center_y + 3), 
                         (120, 180, 160), -1)

# ============================================================================
# STAT SYSTEM
# ============================================================================
@dataclass
class MedicalStats:
    patient_stability: float = 60.0      # Patient health outcomes
    infection_risk: float = 50.0         # Infection control level
    resource_availability: float = 65.0  # Equipment and supplies
    staff_morale: float = 70.0           # Healthcare worker satisfaction
    public_trust: float = 60.0           # Community confidence
    
    # Display stats (smoothed)
    display_patient_stability: float = 60.0
    display_infection_risk: float = 50.0
    display_resource_availability: float = 65.0
    display_staff_morale: float = 70.0
    display_public_trust: float = 60.0
    
    def clamp(self):
        """Ensure all stats remain within 0-100 bounds"""
        self.patient_stability = max(0, min(100, self.patient_stability))
        self.infection_risk = max(0, min(100, self.infection_risk))
        self.resource_availability = max(0, min(100, self.resource_availability))
        self.staff_morale = max(0, min(100, self.staff_morale))
        self.public_trust = max(0, min(100, self.public_trust))
    
    def smooth_update(self, lerp_factor: float = 0.15):
        """Smoothly interpolate display values toward actual values"""
        self.display_patient_stability += (self.patient_stability - self.display_patient_stability) * lerp_factor
        self.display_infection_risk += (self.infection_risk - self.display_infection_risk) * lerp_factor
        self.display_resource_availability += (self.resource_availability - self.display_resource_availability) * lerp_factor
        self.display_staff_morale += (self.staff_morale - self.display_staff_morale) * lerp_factor
        self.display_public_trust += (self.public_trust - self.display_public_trust) * lerp_factor

# ============================================================================
# DECISION SYSTEM
# ============================================================================
class Decision:
    def __init__(self, title: str, description: str, 
                 patient_stability: int, infection_risk: int, 
                 resource_availability: int, staff_morale: int, 
                 public_trust: int):
        self.title = title
        self.description = description
        self.patient_stability = patient_stability
        self.infection_risk = infection_risk
        self.resource_availability = resource_availability
        self.staff_morale = staff_morale
        self.public_trust = public_trust
    
    def apply(self, stats: MedicalStats):
        """Apply decision effects to stats"""
        stats.patient_stability += self.patient_stability
        stats.infection_risk += self.infection_risk
        stats.resource_availability += self.resource_availability
        stats.staff_morale += self.staff_morale
        stats.public_trust += self.public_trust
        stats.clamp()

# ============================================================================
# PHASE DEFINITIONS
# ============================================================================
PHASES = [
    {
        'title': 'PHASE 1: EMERGENCY ROOM OVERLOAD',
        'context': 'Your ER is at 180% capacity. Ambulances are being diverted.\nPatients are waiting 6+ hours. You must act immediately to\nmanage the crisis and prevent deterioration of care quality.',
        'image_key': 'emergency_room',
        'decisions': [
            Decision(
                'Open Overflow Unit',
                'Convert conference rooms and\nrecovery areas into temporary\npatient care spaces',
                patient_stability=8,
                infection_risk=-5,
                resource_availability=-12,
                staff_morale=-8,
                public_trust=6
            ),
            Decision(
                'Deploy Triage Protocol',
                'Implement strict triage system\nto prioritize critical cases and\nredirect non-urgent patients',
                patient_stability=5,
                infection_risk=3,
                resource_availability=8,
                staff_morale=4,
                public_trust=-6
            ),
            Decision(
                'Request Regional Support',
                'Coordinate with nearby hospitals\nto transfer stable patients and\nshare resources',
                patient_stability=3,
                infection_risk=-3,
                resource_availability=6,
                staff_morale=8,
                public_trust=10
            ),
        ],
        'fact': 'Emergency departments operating above 100% capacity see a 5% increase\nin mortality rates and 20% increase in patient wait times. Coordinated\nregional response systems can reduce individual hospital burden by 30%.'
    },
    {
        'title': 'PHASE 2: INFECTION CONTROL STRATEGY',
        'context': 'Hospital-acquired infection rates are rising. You\'ve detected\na cluster of antibiotic-resistant infections in the ICU.\nYou need a comprehensive containment strategy.',
        'image_key': 'infection_control',
        'decisions': [
            Decision(
                'Enhanced Sterilization Protocol',
                'Implement UV-C disinfection and\nquadruple daily cleaning cycles\nacross all high-risk areas',
                patient_stability=6,
                infection_risk=15,
                resource_availability=-15,
                staff_morale=-6,
                public_trust=8
            ),
            Decision(
                'Staff Training Program',
                'Mandatory infection control\ntraining and certification for\nall medical personnel',
                patient_stability=4,
                infection_risk=10,
                resource_availability=-8,
                staff_morale=10,
                public_trust=5
            ),
            Decision(
                'Visitor Restriction Policy',
                'Limit all non-essential visitors\nand implement strict screening\nand PPE requirements',
                patient_stability=8,
                infection_risk=18,
                resource_availability=-4,
                staff_morale=2,
                public_trust=-10
            ),
        ],
        'fact': 'Healthcare-associated infections (HAIs) affect 1 in 31 hospital patients\non any given day. Proper hand hygiene alone can reduce HAIs by 40%.\nComprehensive infection control programs save an average of $6,000 per case.'
    },
    {
        'title': 'PHASE 3: EQUIPMENT ALLOCATION CRISIS',
        'context': 'Budget cuts have forced difficult choices. You must allocate\nlimited funding between competing critical needs.\nYour decision will impact care quality for months.',
        'image_key': 'equipment_allocation',
        'decisions': [
            Decision(
                'Invest in Diagnostic Equipment',
                'Purchase new MRI machine and\nupgrade CT scanners to reduce\ndiagnostic delays',
                patient_stability=12,
                infection_risk=0,
                resource_availability=-20,
                staff_morale=8,
                public_trust=12
            ),
            Decision(
                'Expand ICU Capacity',
                'Add 12 ICU beds with full\nventilator and monitoring\nequipment for critical care',
                patient_stability=15,
                infection_risk=-8,
                resource_availability=-25,
                staff_morale=5,
                public_trust=8
            ),
            Decision(
                'Upgrade IT Systems',
                'Implement electronic health\nrecords and telemedicine\nplatform for efficiency',
                patient_stability=5,
                infection_risk=8,
                resource_availability=10,
                staff_morale=15,
                public_trust=6
            ),
        ],
        'fact': 'Hospitals that invest in modern diagnostic equipment see 30% faster\ndiagnosis times and 15% better patient outcomes. However, equipment costs\nrepresent only 10% of total healthcare spending—staffing is 60%.'
    },
    {
        'title': 'PHASE 4: MEDICAL CRISIS EVENT',
        'context': '',  # Generated dynamically
        'image_key': 'crisis_event',
        'decisions': [],  # Crisis has no decisions - immediate impact
        'fact': ''  # Generated based on crisis type
    },
    {
        'title': 'PHASE 5: LONG-TERM HEALTHCARE POLICY',
        'context': 'Your hospital has weathered the immediate crises. Now you must\nestablish policies that will define care quality and community\nhealth outcomes for the next decade.',
        'image_key': 'long_term_policy',
        'decisions': [
            Decision(
                'Preventive Care Initiative',
                'Launch community health programs\nfor chronic disease management\nand early intervention',
                patient_stability=10,
                infection_risk=12,
                resource_availability=-10,
                staff_morale=12,
                public_trust=20
            ),
            Decision(
                'Specialized Treatment Center',
                'Develop regional center of\nexcellence for cardiac, cancer,\nor neurological care',
                patient_stability=18,
                infection_risk=-5,
                resource_availability=-15,
                staff_morale=8,
                public_trust=15
            ),
            Decision(
                'Staff Development Program',
                'Invest in continuing education,\nmental health support, and\ncareer advancement for staff',
                patient_stability=8,
                infection_risk=6,
                resource_availability=-5,
                staff_morale=25,
                public_trust=10
            ),
        ],
        'fact': 'Investment in preventive care reduces emergency interventions by 25%\nand total healthcare costs by 12-18%. Employee satisfaction directly\ncorrelates with patient outcomes—hospitals with high staff morale have\n15% better patient safety scores.'
    },
]

# ============================================================================
# CRISIS EVENT GENERATION
# ============================================================================
CRISIS_TYPES = [
    {
        'title': '🚨 MASS CASUALTY INCIDENT',
        'context': 'A major highway accident has resulted in 45 casualties\narriving simultaneously. Your ER and trauma teams are\nactivating emergency protocols.',
        'image_key': 'mass_casualty',
        'base_impacts': {
            'patient_stability': -25,
            'infection_risk': -8,
            'resource_availability': -30,
            'staff_morale': -15,
            'public_trust': 5,
        },
        'fact': 'Mass casualty incidents require hospitals to switch from individual\npatient focus to population triage. Hospitals with disaster preparedness\nplans reduce chaos by 60% and improve survival rates by 40%.'
    },
    {
        'title': '🦠 INFECTIOUS DISEASE OUTBREAK',
        'context': 'An unknown respiratory pathogen has been detected in 18 patients.\nThe CDC has been notified. You must contain spread while\nmaintaining normal operations.',
        'image_key': 'outbreak',
        'base_impacts': {
            'patient_stability': -20,
            'infection_risk': -35,
            'resource_availability': -15,
            'staff_morale': -20,
            'public_trust': -15,
        },
        'fact': 'Early detection and isolation of infectious disease outbreaks is\ncritical. Every hour of delay in implementing containment protocols\ncan triple the number of secondary infections.'
    },
    {
        'title': '⚡ POWER GRID FAILURE',
        'context': 'Regional power outage has disabled hospital systems.\nBackup generators are running but fuel is limited to 48 hours.\nCritical equipment must be prioritized.',
        'image_key': 'power_failure',
        'base_impacts': {
            'patient_stability': -18,
            'infection_risk': -10,
            'resource_availability': -25,
            'staff_morale': -12,
            'public_trust': -8,
        },
        'fact': 'Hospital backup power systems support only essential equipment—typically\n40-60% of normal capacity. Life-support systems, ICU, and ER receive\npriority. Modern hospitals maintain 72-96 hour fuel reserves.'
    },
]

# ============================================================================
# MEDICAL DECISION SIMULATION ENGINE
# ============================================================================
class MedicalDecisionSimulation:
    def __init__(self, images_folder: str = "medical_images"):
        self.stats = MedicalStats()
        self.current_phase = 0
        self.state = 'playing'  # 'playing', 'fact_screen', 'reflection'
        self.selected_decision: Optional[Decision] = None
        self.decision_history: List[Tuple[str, Decision]] = []
        
        # Image management
        self.image_manager = ImageManager(images_folder)
        
        # Hover system
        self.hover_target: Optional[int] = None
        self.hover_progress = 0.0
        self.hover_confirm_time = 3.0  # 3 seconds to confirm
        
        # Crisis system
        self.crisis_event: Optional[dict] = None
        self.crisis_already_triggered = False
        
        # Fact screen
        self.current_fact = ""
        self.fact_timer = 0
        self.fact_duration = 180  # 3 seconds at 60fps
        
        # Generate initial crisis
        self._generate_crisis()
    
    def _generate_crisis(self):
        """Generate a random crisis with adjusted impacts based on previous decisions"""
        self.crisis_event = random.choice(CRISIS_TYPES).copy()
        
        # Adjust crisis impact based on preparation
        mitigation_factor = 1.0
        
        # Check if player made good decisions in previous phases
        if len(self.decision_history) >= 2:
            # Good infection control helps with outbreak
            if 'OUTBREAK' in self.crisis_event['title']:
                if self.stats.infection_risk > 65:
                    mitigation_factor = 0.6  # 40% reduction
            
            # Good resource management helps with mass casualty
            if 'CASUALTY' in self.crisis_event['title']:
                if self.stats.resource_availability > 70:
                    mitigation_factor = 0.65  # 35% reduction
            
            # Good staff morale helps with power failure
            if 'POWER' in self.crisis_event['title']:
                if self.stats.staff_morale > 75:
                    mitigation_factor = 0.7  # 30% reduction
        
        # Apply mitigation
        for key in self.crisis_event['base_impacts']:
            self.crisis_event['base_impacts'][key] = int(
                self.crisis_event['base_impacts'][key] * mitigation_factor
            )
    
    def reset(self):
        """Reset simulation to initial state"""
        self.stats = MedicalStats()
        self.current_phase = 0
        self.state = 'playing'
        self.selected_decision = None
        self.decision_history = []
        self.hover_target = None
        self.hover_progress = 0.0
        self.crisis_already_triggered = False
        self.crisis_event = None
        self._generate_crisis()
    
    def evaluate(self) -> dict:
        """Evaluate final performance and generate analysis"""
        avg_score = (
            self.stats.patient_stability +
            self.stats.infection_risk +
            self.stats.resource_availability +
            self.stats.staff_morale +
            self.stats.public_trust
        ) / 5.0
        
        # Determine grade and analysis
        if avg_score >= 80:
            grade = "EXCEPTIONAL LEADERSHIP"
            analysis = "Your hospital management demonstrates outstanding competence.\nYou maintained excellent patient care while building community\ntrust and supporting your healthcare team through multiple crises."
            color = COLORS['stat_excellent']
        elif avg_score >= 65:
            grade = "COMPETENT ADMINISTRATION"
            analysis = "You successfully navigated challenging medical scenarios.\nYour hospital maintained operational effectiveness despite\nsignificant pressure. Some areas show room for improvement."
            color = COLORS['stat_good']
        elif avg_score >= 50:
            grade = "ADEQUATE RESPONSE"
            analysis = "Your hospital survived critical challenges but faced difficulties.\nSeveral key metrics show concerning trends. Strategic improvements\nare needed to ensure sustainable healthcare delivery."
            color = COLORS['stat_warning']
        else:
            grade = "CRITICAL DEFICIENCIES"
            analysis = "Hospital operations faced severe strain across multiple areas.\nPatient outcomes, staff wellbeing, or community trust suffered\nsignificantly. Immediate intervention required."
            color = COLORS['stat_critical']
        
        return {
            'grade': grade,
            'analysis': analysis,
            'color': color,
            'avg_score': avg_score
        }
    
    def update(self, dt: float = 1/60):
        """Update simulation state"""
        # Smooth stat display values
        self.stats.smooth_update()
        
        # Update fact screen timer
        if self.state == 'fact_screen':
            self.fact_timer += 1
            if self.fact_timer >= self.fact_duration:
                self.fact_timer = 0
                self.current_phase += 1
                
                # Check if simulation complete
                if self.current_phase >= len(PHASES):
                    self.state = 'reflection'
                else:
                    self.state = 'playing'
                    
                    # Handle crisis phase
                    if PHASES[self.current_phase]['title'].startswith('PHASE 4'):
                        if not self.crisis_already_triggered:
                            self._trigger_crisis()
    
    def _trigger_crisis(self):
        """Apply crisis impacts and mark as triggered"""
        if self.crisis_event:
            self.stats.patient_stability += self.crisis_event['base_impacts']['patient_stability']
            self.stats.infection_risk += self.crisis_event['base_impacts']['infection_risk']
            self.stats.resource_availability += self.crisis_event['base_impacts']['resource_availability']
            self.stats.staff_morale += self.crisis_event['base_impacts']['staff_morale']
            self.stats.public_trust += self.crisis_event['base_impacts']['public_trust']
            self.stats.clamp()
            
            # Update phase context and image
            PHASES[self.current_phase]['context'] = self.crisis_event['context']
            PHASES[self.current_phase]['fact'] = self.crisis_event['fact']
            PHASES[self.current_phase]['image_key'] = self.crisis_event['image_key']
            
            self.crisis_already_triggered = True
            
            # Auto-advance after crisis display
            self.current_fact = self.crisis_event['fact']
            self.state = 'fact_screen'
            self.fact_timer = 0
    
    def make_decision(self, decision_idx: int):
        """Execute a decision"""
        phase = PHASES[self.current_phase]
        
        if decision_idx < len(phase['decisions']):
            decision = phase['decisions'][decision_idx]
            decision.apply(self.stats)
            
            # Record decision
            self.decision_history.append((phase['title'], decision))
            self.selected_decision = decision
            
            # Show fact screen
            self.current_fact = phase['fact']
            self.state = 'fact_screen'
            self.fact_timer = 0

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def alpha_blend(bg: np.ndarray, fg: np.ndarray, alpha: float):
    """Blend foreground onto background with alpha transparency"""
    return cv2.addWeighted(bg, 1 - alpha, fg, alpha, 0)

def wrap_text(text: str, max_width: int, font, font_scale: float, thickness: int) -> List[str]:
    """Wrap text to fit within max_width"""
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        
        if w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines

def draw_stat_bar(frame: np.ndarray, x: int, y: int, width: int, height: int,
                  label: str, value: float, max_value: float = 100):
    """Draw a professional stat bar with gradient and label"""
    # Determine color based on value
    if value >= 75:
        bar_color = COLORS['stat_excellent']
    elif value >= 60:
        bar_color = COLORS['stat_good']
    elif value >= 40:
        bar_color = COLORS['stat_warning']
    else:
        bar_color = COLORS['stat_critical']
    
    # Background bar
    cv2.rectangle(frame, (x, y), (x + width, y + height), COLORS['bg_panel'], -1)
    
    # Filled portion
    fill_width = int((value / max_value) * width)
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), bar_color, -1)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), COLORS['text_dim'], 2)
    
    # Label
    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, COLORS['text_secondary'], 1, cv2.LINE_AA)
    
    # Value text
    value_text = f"{int(value)}"
    (tw, th), _ = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, value_text, (x + width + 10, y + height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_primary'], 1, cv2.LINE_AA)

def point_in_rect(px: int, py: int, rx: int, ry: int, rw: int, rh: int) -> bool:
    """Check if point is inside rectangle"""
    return rx <= px <= rx + rw and ry <= py <= ry + rh

# ============================================================================
# MAIN RENDERING FUNCTION
# ============================================================================
def run_medical(frame: np.ndarray, hand_pos: Optional[Tuple[int, int]], 
                simulation: MedicalDecisionSimulation) -> np.ndarray:
    """
    Main rendering function for medical decision simulation
    
    Args:
        frame: Input video frame (1280x720)
        hand_pos: (x, y) hand tracking position or None
        simulation: MedicalDecisionSimulation instance
    
    Returns:
        Rendered frame with UI overlay
    """
    h, w = frame.shape[:2]
    
    # Update simulation
    simulation.update()
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    
    # ========================================================================
    # REFLECTION SCREEN
    # ========================================================================
    if simulation.state == 'reflection':
        # Dark background
        cv2.rectangle(overlay, (0, 0), (w, h), COLORS['bg_primary'], -1)
        
        # Evaluate performance
        eval_result = simulation.evaluate()
        
        # Header
        cv2.putText(overlay, "HOSPITAL ADMINISTRATION REVIEW", (w//2 - 280, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS['text_primary'], 2, cv2.LINE_AA)
        
        # Grade panel
        grade_panel_y = 140
        grade_panel_h = 120
        cv2.rectangle(overlay, (w//2 - 300, grade_panel_y), (w//2 + 300, grade_panel_y + grade_panel_h),
                      COLORS['bg_secondary'], -1)
        cv2.rectangle(overlay, (w//2 - 300, grade_panel_y), (w//2 + 300, grade_panel_y + grade_panel_h),
                      eval_result['color'], 3)
        
        cv2.putText(overlay, eval_result['grade'], (w//2 - 250, grade_panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, eval_result['color'], 2, cv2.LINE_AA)
        
        cv2.putText(overlay, f"Overall Performance: {int(eval_result['avg_score'])}/100",
                    (w//2 - 250, grade_panel_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text_secondary'], 1, cv2.LINE_AA)
        
        # Analysis panel
        analysis_y = grade_panel_y + grade_panel_h + 30
        analysis_lines = wrap_text(eval_result['analysis'], 580, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        cv2.rectangle(overlay, (w//2 - 300, analysis_y), (w//2 + 300, analysis_y + 140),
                      COLORS['bg_panel'], -1)
        
        for i, line in enumerate(analysis_lines):
            cv2.putText(overlay, line, (w//2 - 280, analysis_y + 35 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text_primary'], 1, cv2.LINE_AA)
        
        # Final stats
        stats_y = analysis_y + 160
        cv2.putText(overlay, "FINAL METRICS", (w//2 - 280, stats_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text_secondary'], 1, cv2.LINE_AA)
        
        stat_labels = [
            ("Patient Stability", simulation.stats.patient_stability),
            ("Infection Control", simulation.stats.infection_risk),
            ("Resource Status", simulation.stats.resource_availability),
            ("Staff Morale", simulation.stats.staff_morale),
            ("Public Trust", simulation.stats.public_trust),
        ]
        
        for i, (label, value) in enumerate(stat_labels):
            draw_stat_bar(overlay, w//2 - 280, stats_y + 30 + i * 35, 250, 20, label, value)
        
        # Restart button
        restart_btn_x = w//2 - 100
        restart_btn_y = h - 100
        restart_btn_w = 200
        restart_btn_h = 50
        
        btn_hover = False
        if hand_pos:
            btn_hover = point_in_rect(hand_pos[0], hand_pos[1], 
                                      restart_btn_x, restart_btn_y, restart_btn_w, restart_btn_h)
        
        btn_color = COLORS['hover_overlay'] if btn_hover else COLORS['accent_blue']
        cv2.rectangle(overlay, (restart_btn_x, restart_btn_y), 
                      (restart_btn_x + restart_btn_w, restart_btn_y + restart_btn_h),
                      btn_color, -1)
        cv2.rectangle(overlay, (restart_btn_x, restart_btn_y), 
                      (restart_btn_x + restart_btn_w, restart_btn_y + restart_btn_h),
                      COLORS['text_primary'], 2)
        
        cv2.putText(overlay, "RESTART", (restart_btn_x + 45, restart_btn_y + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text_primary'], 2, cv2.LINE_AA)
        
        # Handle restart
        if btn_hover:
            simulation.hover_progress += 1/60 / simulation.hover_confirm_time
            if simulation.hover_progress >= 1.0:
                simulation.reset()
                simulation.hover_progress = 0.0
        else:
            simulation.hover_progress = 0.0
        
        return alpha_blend(frame, overlay, 0.92)
    
    # ========================================================================
    # FACT SCREEN
    # ========================================================================
    if simulation.state == 'fact_screen':
        # Dark background
        cv2.rectangle(overlay, (0, 0), (w, h), COLORS['bg_primary'], -1)
        
        # Header
        cv2.putText(overlay, "MEDICAL INSIGHT", (w//2 - 150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS['accent_teal'], 2, cv2.LINE_AA)
        
        # Fact panel
        fact_panel_x = 150
        fact_panel_y = 180
        fact_panel_w = w - 300
        fact_panel_h = 300
        
        cv2.rectangle(overlay, (fact_panel_x, fact_panel_y),
                      (fact_panel_x + fact_panel_w, fact_panel_y + fact_panel_h),
                      COLORS['bg_secondary'], -1)
        cv2.rectangle(overlay, (fact_panel_x, fact_panel_y),
                      (fact_panel_x + fact_panel_w, fact_panel_y + fact_panel_h),
                      COLORS['accent_teal'], 3)
        
        # Wrap and display fact text
        fact_lines = simulation.current_fact.split('\n')
        wrapped_lines = []
        for line in fact_lines:
            wrapped_lines.extend(wrap_text(line, fact_panel_w - 60, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1))
        
        for i, line in enumerate(wrapped_lines):
            cv2.putText(overlay, line, (fact_panel_x + 30, fact_panel_y + 50 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text_primary'], 1, cv2.LINE_AA)
        
        # Progress indicator
        progress = simulation.fact_timer / simulation.fact_duration
        progress_w = int((w - 400) * progress)
        cv2.rectangle(overlay, (200, h - 80), (200 + progress_w, h - 60),
                      COLORS['accent_teal'], -1)
        cv2.rectangle(overlay, (200, h - 80), (w - 200, h - 60),
                      COLORS['text_dim'], 2)
        
        return alpha_blend(frame, overlay, 0.92)
    
    # ========================================================================
    # MAIN GAME SCREEN
    # ========================================================================
    
    # Header panel
    header_h = 90
    cv2.rectangle(overlay, (0, 0), (w, header_h), COLORS['bg_secondary'], -1)
    cv2.line(overlay, (0, header_h), (w, header_h), COLORS['accent_blue'], 3)
    
    # Title
    cv2.putText(overlay, "HOSPITAL ADMINISTRATION COMMAND", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLORS['text_primary'], 2, cv2.LINE_AA)
    
    # Phase indicator
    phase = PHASES[simulation.current_phase]
    cv2.putText(overlay, phase['title'], (30, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['accent_teal'], 1, cv2.LINE_AA)
    
    # ========================================================================
    # THREE COLUMN LAYOUT
    # ========================================================================
    
    col_margin = 20
    col_y = header_h + 20
    col_h = h - header_h - 240  # Leave room for decision cards
    
    # Column widths
    col1_w = 320  # Stats
    col2_w = 450  # Context/Image
    col3_w = 380  # Preview
    
    col1_x = col_margin
    col2_x = col1_x + col1_w + col_margin
    col3_x = col2_x + col2_w + col_margin
    
    # ========================================================================
    # COLUMN 1: STATS
    # ========================================================================
    cv2.rectangle(overlay, (col1_x, col_y), (col1_x + col1_w, col_y + col_h),
                  COLORS['bg_panel'], -1)
    cv2.rectangle(overlay, (col1_x, col_y), (col1_x + col1_w, col_y + col_h),
                  COLORS['text_dim'], 2)
    
    cv2.putText(overlay, "HOSPITAL METRICS", (col1_x + 15, col_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text_primary'], 1, cv2.LINE_AA)
    
    # Stat bars
    stat_y = col_y + 60
    stat_spacing = 70
    stat_bar_w = 220
    
    draw_stat_bar(overlay, col1_x + 20, stat_y, stat_bar_w, 24,
                  "Patient Stability", simulation.stats.display_patient_stability)
    
    draw_stat_bar(overlay, col1_x + 20, stat_y + stat_spacing, stat_bar_w, 24,
                  "Infection Control", simulation.stats.display_infection_risk)
    
    draw_stat_bar(overlay, col1_x + 20, stat_y + stat_spacing * 2, stat_bar_w, 24,
                  "Resource Status", simulation.stats.display_resource_availability)
    
    draw_stat_bar(overlay, col1_x + 20, stat_y + stat_spacing * 3, stat_bar_w, 24,
                  "Staff Morale", simulation.stats.display_staff_morale)
    
    draw_stat_bar(overlay, col1_x + 20, stat_y + stat_spacing * 4, stat_bar_w, 24,
                  "Public Trust", simulation.stats.display_public_trust)
    
    # ========================================================================
    # COLUMN 2: CONTEXT & IMAGE
    # ========================================================================
    cv2.rectangle(overlay, (col2_x, col_y), (col2_x + col2_w, col_y + col_h),
                  COLORS['bg_panel'], -1)
    cv2.rectangle(overlay, (col2_x, col_y), (col2_x + col2_w, col_y + col_h),
                  COLORS['text_dim'], 2)
    
    # Context text
    context_lines = phase['context'].split('\n')
    for i, line in enumerate(context_lines):
        cv2.putText(overlay, line, (col2_x + 20, col_y + 35 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['text_primary'], 1, cv2.LINE_AA)
    
    # Load and display image
    img_y = col_y + 160
    img_h = 200
    img_w = col2_w - 40
    
    scenario_image = simulation.image_manager.load_image(phase['image_key'], img_w, img_h)
    
    # Place image on overlay
    overlay[img_y:img_y + img_h, col2_x + 20:col2_x + 20 + img_w] = scenario_image
    
    # Border around image
    cv2.rectangle(overlay, (col2_x + 20, img_y), (col2_x + 20 + img_w, img_y + img_h),
                  COLORS['accent_blue'], 2)
    
    # ========================================================================
    # COLUMN 3: DECISION PREVIEW
    # ========================================================================
    cv2.rectangle(overlay, (col3_x, col_y), (col3_x + col3_w, col_y + col_h),
                  COLORS['bg_panel'], -1)
    cv2.rectangle(overlay, (col3_x, col_y), (col3_x + col3_w, col_y + col_h),
                  COLORS['text_dim'], 2)
    
    cv2.putText(overlay, "DECISION PREVIEW", (col3_x + 15, col_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text_primary'], 1, cv2.LINE_AA)
    
    # Show preview if hovering over decision
    if simulation.hover_target is not None and len(phase['decisions']) > 0:
        if simulation.hover_target < len(phase['decisions']):
            decision = phase['decisions'][simulation.hover_target]
            
            preview_y = col_y + 60
            
            cv2.putText(overlay, decision.title, (col3_x + 20, preview_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS['accent_teal'], 1, cv2.LINE_AA)
            
            # Impact preview
            impacts = [
                ("Patient Stability", decision.patient_stability, COLORS['stat_good'] if decision.patient_stability >= 0 else COLORS['stat_critical']),
                ("Infection Risk", decision.infection_risk, COLORS['stat_good'] if decision.infection_risk >= 0 else COLORS['stat_critical']),
                ("Resources", decision.resource_availability, COLORS['stat_good'] if decision.resource_availability >= 0 else COLORS['stat_critical']),
                ("Staff Morale", decision.staff_morale, COLORS['stat_good'] if decision.staff_morale >= 0 else COLORS['stat_critical']),
                ("Public Trust", decision.public_trust, COLORS['stat_good'] if decision.public_trust >= 0 else COLORS['stat_critical']),
            ]
            
            impact_y = preview_y + 40
            for i, (label, value, color) in enumerate(impacts):
                sign = "+" if value >= 0 else ""
                cv2.putText(overlay, f"{label}:", (col3_x + 20, impact_y + i * 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_secondary'], 1, cv2.LINE_AA)
                cv2.putText(overlay, f"{sign}{value}", (col3_x + 220, impact_y + i * 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    else:
        cv2.putText(overlay, "Hover over a decision", (col3_x + 20, col_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['text_dim'], 1, cv2.LINE_AA)
        cv2.putText(overlay, "to see impact preview", (col3_x + 20, col_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['text_dim'], 1, cv2.LINE_AA)
    
    # ========================================================================
    # DECISION CARDS (Bottom)
    # ========================================================================
    
    # Crisis phase has no decisions - show automatic event
    if phase['title'].startswith('PHASE 4') and not simulation.crisis_already_triggered:
        crisis_card_y = h - 200
        crisis_card_h = 170
        
        cv2.rectangle(overlay, (col_margin, crisis_card_y),
                      (w - col_margin, crisis_card_y + crisis_card_h),
                      COLORS['bg_secondary'], -1)
        cv2.rectangle(overlay, (col_margin, crisis_card_y),
                      (w - col_margin, crisis_card_y + crisis_card_h),
                      COLORS['accent_red'], 4)
        
        if simulation.crisis_event:
            cv2.putText(overlay, simulation.crisis_event['title'], (col_margin + 30, crisis_card_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS['accent_red'], 2, cv2.LINE_AA)
            
            context_lines = simulation.crisis_event['context'].split('\n')
            for i, line in enumerate(context_lines):
                cv2.putText(overlay, line, (col_margin + 30, crisis_card_y + 80 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['text_primary'], 1, cv2.LINE_AA)
    
    elif len(phase['decisions']) > 0:
        # Normal decision cards
        decision_y = h - 200
        decision_h = 170
        decision_w = (w - col_margin * 4) // 3
        
        for i, decision in enumerate(phase['decisions']):
            decision_x = col_margin + i * (decision_w + col_margin)
            
            # Check hover
            is_hovering = False
            if hand_pos:
                is_hovering = point_in_rect(hand_pos[0], hand_pos[1],
                                           decision_x, decision_y, decision_w, decision_h)
            
            # Update hover state
            if is_hovering:
                if simulation.hover_target != i:
                    simulation.hover_target = i
                    simulation.hover_progress = 0.0
                else:
                    simulation.hover_progress += 1/60 / simulation.hover_confirm_time
                    if simulation.hover_progress >= 1.0:
                        simulation.make_decision(i)
                        simulation.hover_progress = 0.0
                        simulation.hover_target = None
            else:
                if simulation.hover_target == i:
                    simulation.hover_target = None
                    simulation.hover_progress = 0.0
            
            # Card background
            card_color = COLORS['hover_overlay'] if is_hovering else COLORS['bg_secondary']
            cv2.rectangle(overlay, (decision_x, decision_y),
                          (decision_x + decision_w, decision_y + decision_h),
                          card_color, -1)
            cv2.rectangle(overlay, (decision_x, decision_y),
                          (decision_x + decision_w, decision_y + decision_h),
                          COLORS['accent_blue'], 3)
            
            # Title
            title_lines = wrap_text(decision.title, decision_w - 30, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
            for j, line in enumerate(title_lines[:2]):  # Max 2 lines
                cv2.putText(overlay, line, (decision_x + 15, decision_y + 35 + j * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS['text_primary'], 1, cv2.LINE_AA)
            
            # Description
            desc_lines = wrap_text(decision.description, decision_w - 30, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            for j, line in enumerate(desc_lines[:3]):  # Max 3 lines
                cv2.putText(overlay, line, (decision_x + 15, decision_y + 85 + j * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text_secondary'], 1, cv2.LINE_AA)
            
            # Hover progress bar
            if is_hovering and simulation.hover_progress > 0:
                progress_w = int((decision_w - 20) * simulation.hover_progress)
                cv2.rectangle(overlay, (decision_x + 10, decision_y + decision_h - 15),
                              (decision_x + 10 + progress_w, decision_y + decision_h - 5),
                              COLORS['accent_teal'], -1)
    
    # Hand cursor
    if hand_pos:
        cv2.circle(overlay, hand_pos, 12, COLORS['accent_teal'], -1)
        cv2.circle(overlay, hand_pos, 12, COLORS['text_primary'], 2)
    
    # Blend overlay
    return alpha_blend(frame, overlay, 0.88)

# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Initialize simulation
    sim = MedicalDecisionSimulation()
    
    # Demo with mock video feed
    print("Medical Decision Simulation - Professional High Visibility Edition")
    print("Initialized successfully with image support.")
    print(f"Current phase: {PHASES[sim.current_phase]['title']}")
    print(f"Stats: Patient={sim.stats.patient_stability:.1f}, Infection={sim.stats.infection_risk:.1f}")