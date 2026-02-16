"""
Finance Simulation - PROFESSIONAL EDITION
Sophisticated UI with smooth animations, elegant color scheme, and dynamic interactions
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time
import random
import math

@dataclass
class FinancialStats:
    stability: float = 50.0
    risk: float = 30.0
    liquidity: float = 40.0
    credit_score: float = 50.0
    knowledge: float = 0.0
    
    def clamp(self):
        self.stability = max(0, min(100, self.stability))
        self.risk = max(0, min(100, self.risk))
        self.liquidity = max(0, min(100, self.liquidity))
        self.credit_score = max(0, min(100, self.credit_score))
        self.knowledge = max(0, min(100, self.knowledge))

@dataclass
class Decision:
    title: str
    impact: Dict[str, float]
    description: str

class AnimatedValue:
    """Smooth value animation with easing"""
    def __init__(self, initial: float = 0.0, speed: float = 0.08):
        self.current = initial
        self.target = initial
        self.speed = speed
    
    def set_target(self, value: float):
        self.target = value
    
    def update(self):
        diff = self.target - self.current
        self.current += diff * self.speed
        return self.current
    
    def get(self) -> float:
        return self.current

class FinanceSimulation:
    def __init__(self):
        self.stats = FinancialStats()
        self.target_stats = FinancialStats()
        
        # Animated stats for smooth transitions
        self.animated_stats = {
            'stability': AnimatedValue(50.0, 0.06),
            'risk': AnimatedValue(30.0, 0.06),
            'liquidity': AnimatedValue(40.0, 0.06),
            'credit_score': AnimatedValue(50.0, 0.06),
            'knowledge': AnimatedValue(0.0, 0.06)
        }
        
        self.phase = 0
        self.active = False
        self.completed = False
        
        self.hovered_button = -1
        self.hover_start_time = 0
        self.hover_duration = 1.0
        self.showing_fact = False
        self.fact_display_time = 0
        self.fact_duration = 3.0
        self.current_fact = ""
        
        self.decision_history = []
        self.insurance_level = 0
        self.crisis_outcome = ""
        
        # Animation state
        self.time = 0.0
        self.phase_transition_alpha = AnimatedValue(0.0, 0.12)
        self.transitioning_phase = False
        self.button_hover_alpha = [AnimatedValue(0.0, 0.15) for _ in range(4)]
        self.stat_bar_values = {
            'stability': AnimatedValue(50.0, 0.05),
            'risk': AnimatedValue(30.0, 0.05),
            'liquidity': AnimatedValue(40.0, 0.05),
            'credit_score': AnimatedValue(50.0, 0.05)
        }
        self.particle_effects = []
        
        # Professional color palette
        self.colors = {
            'bg_primary': (12, 18, 28),          # Deep blue-black
            'bg_secondary': (18, 26, 42),        # Slightly lighter
            'bg_card': (24, 32, 48),             # Card backgrounds
            'accent_primary': (88, 166, 255),    # Refined blue
            'accent_gold': (255, 193, 102),      # Sophisticated gold
            'accent_green': (102, 255, 178),     # Mint green
            'accent_red': (255, 112, 112),       # Coral red
            'text_primary': (240, 246, 252),     # Almost white
            'text_secondary': (148, 163, 184),   # Muted blue-gray
            'text_tertiary': (100, 116, 139),    # Darker gray
            'border_subtle': (51, 65, 85),       # Subtle borders
            'border_bright': (71, 85, 105),      # Brighter borders
            'glow_blue': (59, 130, 246),         # Glow effect
            'glow_gold': (251, 191, 36),         # Gold glow
        }
        
        self.images = {}
        self.load_images()
        self.phases = self.define_phases()
        
    def load_images(self):
        image_files = {
            'housing': 'assets/housing.png',
            'insurance': 'assets/insurance.png',
            'credit_card': 'assets/credit_card.png',
            'crisis': 'assets/crisis.png',
            'investment': 'assets/investment.png'
        }
        
        for key, path in image_files.items():
            try:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    img = cv2.resize(img, (320, 320))
                    self.images[key] = img
                else:
                    self.images[key] = self.create_placeholder(key)
            except:
                self.images[key] = self.create_placeholder(key)
    
    def create_placeholder(self, label: str) -> np.ndarray:
        """Create elegant placeholder with gradient"""
        img = np.zeros((320, 320, 4), dtype=np.uint8)
        img[:, :, 3] = 200
        
        # Gradient background
        for i in range(320):
            intensity = int(40 + (i / 320) * 30)
            img[i, :, 0] = intensity
            img[i, :, 1] = intensity + 10
            img[i, :, 2] = intensity + 20
        
        # Icon representation
        center_x, center_y = 160, 160
        cv2.circle(img, (center_x, center_y), 60, self.colors['accent_primary'], 2)
        cv2.circle(img, (center_x, center_y), 40, self.colors['accent_primary'], 1)
        
        return img
    
    def define_phases(self) -> List[Dict]:
        return [
            {
                'title': 'Housing Commitment',
                'subtitle': 'Foundation of Financial Stability',
                'context': 'Your living situation significantly impacts monthly cash flow and long-term wealth accumulation.',
                'image': 'housing',
                'decisions': [
                    Decision('Luxury Apartment', 
                            {'stability': -15, 'risk': 15, 'liquidity': -25, 'credit_score': 10},
                            'Premium location, high monthly commitment'),
                    Decision('Shared Living',
                            {'stability': 10, 'risk': -5, 'liquidity': 10, 'credit_score': 5},
                            'Balanced approach with flexibility'),
                    Decision('Family Support',
                            {'stability': 15, 'risk': -10, 'liquidity': 25, 'credit_score': 0},
                            'Maximize savings potential')
                ],
                'fact': 'Financial advisors recommend keeping housing costs below 30% of gross income for optimal wealth building.'
            },
            {
                'title': 'Insurance Strategy',
                'subtitle': 'Protection Against Uncertainty',
                'context': 'Insurance is an investment in resilience, trading predictable costs for unpredictable protection.',
                'image': 'insurance',
                'decisions': [
                    Decision('Comprehensive Coverage',
                            {'stability': 20, 'risk': -20, 'liquidity': -15, 'credit_score': 10},
                            'Full protection across all major risks'),
                    Decision('Essential Coverage',
                            {'stability': 10, 'risk': -5, 'liquidity': -5, 'credit_score': 5},
                            'Core protection only'),
                    Decision('Self-Insurance',
                            {'stability': -10, 'risk': 25, 'liquidity': 5, 'credit_score': -5},
                            'Accept all risks personally')
                ],
                'fact': 'Medical emergencies are the leading cause of personal bankruptcy, accounting for over 60% of cases in developed nations.'
            },
            {
                'title': 'Lifestyle Calibration',
                'subtitle': 'Consumption vs. Capital',
                'context': 'Every dollar spent today is a dollar that cannot compound tomorrow. Balance is key.',
                'image': 'credit_card',
                'decisions': [
                    Decision('Leveraged Lifestyle',
                            {'stability': -20, 'risk': 30, 'liquidity': 10, 'credit_score': -15},
                            'Finance present consumption, defer costs'),
                    Decision('Mindful Spending',
                            {'stability': 15, 'risk': -10, 'liquidity': 5, 'credit_score': 10},
                            'Strategic allocation with discipline'),
                    Decision('Extreme Frugality',
                            {'stability': 25, 'risk': -15, 'liquidity': 20, 'credit_score': 5},
                            'Minimize consumption, maximize savings')
                ],
                'fact': 'Consumer debt compounds at rates often 5-10x higher than investment returns, creating a wealth destruction cycle.'
            },
            {
                'title': 'Crisis Response',
                'subtitle': 'Test of Preparedness',
                'context': None,
                'image': 'crisis',
                'decisions': [],
                'fact': 'Studies show that individuals with 6 months of expenses saved are 3x more likely to recover from financial shocks without lasting damage.'
            },
            {
                'title': 'Capital Deployment',
                'subtitle': 'Growth vs. Security',
                'context': 'Investment decisions should align with your risk capacity, time horizon, and financial foundation.',
                'image': 'investment',
                'decisions': [
                    Decision('Venture Capital',
                            {'stability': 0, 'risk': 25, 'liquidity': -20, 'credit_score': 0},
                            'High-risk, high-reward opportunity'),
                    Decision('Fixed Income',
                            {'stability': 15, 'risk': -10, 'liquidity': -10, 'credit_score': 10},
                            'Guaranteed returns, capital preservation'),
                    Decision('Cash Position',
                            {'stability': -5, 'risk': 0, 'liquidity': 5, 'credit_score': 0},
                            'Maintain maximum optionality')
                ],
                'fact': 'Diversified portfolios historically reduce volatility by 40% while maintaining 90% of returns compared to concentrated positions.'
            }
        ]
    
    def generate_crisis(self) -> Dict:
        crises = [
            {'type': 'Medical Emergency', 'desc': 'Unexpected hospitalization requiring immediate financial response'},
            {'type': 'Employment Disruption', 'desc': 'Sudden income loss requiring 3-month bridge period'},
            {'type': 'Critical Asset Failure', 'desc': 'Essential equipment breakdown demanding immediate replacement'}
        ]
        
        crisis = random.choice(crises)
        
        base_stability_loss = 25
        base_liquidity_loss = 30
        base_risk_increase = 20
        
        if self.insurance_level == 2:
            base_stability_loss *= 0.3
            base_liquidity_loss *= 0.4
            base_risk_increase *= 0.2
            crisis['outcome'] = 'Insurance coverage absorbed majority of impact. Minimal disruption to financial position.'
        elif self.insurance_level == 1:
            base_stability_loss *= 0.6
            base_liquidity_loss *= 0.7
            base_risk_increase *= 0.5
            crisis['outcome'] = 'Partial coverage provided. Significant out-of-pocket expenses required.'
        else:
            crisis['outcome'] = 'Full financial impact absorbed directly. Substantial setback to financial goals.'
        
        if self.stats.liquidity > 60:
            base_stability_loss *= 0.7
            crisis['outcome'] += ' Strong liquidity cushioned the blow.'
        elif self.stats.liquidity < 30:
            base_stability_loss *= 1.3
            base_risk_increase *= 1.2
            crisis['outcome'] += ' Insufficient reserves amplified the impact.'
        
        crisis['impact'] = {
            'stability': -base_stability_loss,
            'risk': base_risk_increase,
            'liquidity': -base_liquidity_loss,
            'credit_score': -10
        }
        
        self.crisis_outcome = crisis['outcome']
        return crisis
    
    def calculate_investment_outcome(self, decision_idx: int):
        decision = self.phases[4]['decisions'][decision_idx]
        
        if decision_idx == 0:
            success_threshold = 40 + (self.stats.stability * 0.3) + (self.stats.credit_score * 0.2)
            success_threshold -= (self.stats.risk * 0.3)
            
            if random.random() * 100 < success_threshold:
                decision.impact = {'stability': 20, 'risk': -10, 'liquidity': 30, 'credit_score': 15}
                self.current_fact = 'Investment succeeded! Strong fundamentals enabled capital growth. Diversification remains key to long-term wealth preservation.'
            else:
                decision.impact = {'stability': -25, 'risk': 15, 'liquidity': -20, 'credit_score': -10}
                self.current_fact = 'Investment underperformed. Weak financial foundation amplified losses. Only invest capital you can afford to lose entirely.'
    
    def run_finance(self, frame: np.ndarray, hand_pos: Optional[Tuple[int, int]]) -> np.ndarray:
        if not self.active:
            return frame
        
        self.time += 0.016  # ~60fps
        self.update_animations()
        
        if self.showing_fact:
            frame = self.render_fact_screen(frame)
        elif self.completed:
            frame = self.render_reflection(frame)
        else:
            frame = self.render_phase(frame, hand_pos)
        
        return frame
    
    def update_animations(self):
        # Update all animated values
        for key, anim in self.animated_stats.items():
            target = getattr(self.target_stats, key)
            anim.set_target(target)
            anim.update()
        
        for key, anim in self.stat_bar_values.items():
            anim.update()
        
        for anim in self.button_hover_alpha:
            anim.update()
        
        self.phase_transition_alpha.update()
    
    def render_phase(self, frame: np.ndarray, hand_pos: Optional[Tuple[int, int]]) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Elegant gradient overlay
        overlay = self.create_gradient_overlay(w, h)
        frame = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
        
        # Animated background particles
        self.render_ambient_particles(frame, w, h)
        
        # Generate crisis
        if self.phase == 3 and not self.phases[3]['decisions']:
            crisis = self.generate_crisis()
            self.phases[3]['context'] = crisis['desc']
            self.phases[3]['decisions'] = [
                Decision('Accept Impact', crisis['impact'], crisis['outcome'])
            ]
        
        phase_data = self.phases[self.phase]
        
        # Header with sophisticated typography
        self.render_header(frame, w, phase_data['title'], phase_data['subtitle'])
        
        # Three-column layout
        col_width = w // 3
        
        # Left: Stats panel with animated bars
        self.render_stats_panel_elegant(frame, 40, 140, col_width - 80)
        
        # Center: Context and image
        center_x = col_width + 40
        if phase_data['context']:
            self.render_context_elegant(frame, center_x, 140, col_width - 80, phase_data['context'])
        
        # Image below context
        if phase_data['image'] in self.images:
            img_y = 280
            self.render_image_elegant(frame, center_x + (col_width - 80 - 320)//2, img_y, phase_data['image'])
        
        # Right: Preview or info
        right_x = 2 * col_width + 40
        if self.hovered_button >= 0 and self.hovered_button < len(phase_data['decisions']):
            self.render_preview_elegant(frame, right_x, 140, col_width - 80, phase_data['decisions'][self.hovered_button])
        else:
            self.render_phase_info(frame, right_x, 140, col_width - 80)
        
        # Decision cards at bottom
        button_y = h - 180
        decisions = phase_data['decisions']
        
        if len(decisions) > 0:
            card_width = 320
            card_spacing = 40
            total_width = len(decisions) * card_width + (len(decisions) - 1) * card_spacing
            start_x = (w - total_width) // 2
            
            for i, decision in enumerate(decisions):
                card_x = start_x + i * (card_width + card_spacing)
                is_hovered = self.check_hover(hand_pos, card_x, button_y, card_width, 150)
                
                self.render_decision_card(frame, card_x, button_y, card_width, 150,
                                         decision, i, is_hovered, hand_pos)
        
        return frame
    
    def create_gradient_overlay(self, w: int, h: int) -> np.ndarray:
        """Create sophisticated gradient overlay"""
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Radial gradient from center
        center_x, center_y = w // 2, h // 3
        for y in range(h):
            for x in range(w):
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = math.sqrt(center_x**2 + center_y**2)
                factor = min(1.0, dist / max_dist)
                
                color = (
                    int(self.colors['bg_primary'][0] * (1 - factor*0.3)),
                    int(self.colors['bg_primary'][1] * (1 - factor*0.3)),
                    int(self.colors['bg_primary'][2] * (1 - factor*0.3))
                )
                overlay[y, x] = color
        
        return overlay
    
    def render_ambient_particles(self, frame: np.ndarray, w: int, h: int):
        """Subtle floating particles for depth"""
        num_particles = 15
        for i in range(num_particles):
            seed = i + self.time * 0.1
            x = int((math.sin(seed * 0.5) * 0.5 + 0.5) * w)
            y = int((math.cos(seed * 0.3) * 0.5 + 0.5) * h)
            size = int(2 + math.sin(seed) * 1)
            alpha = 0.15 + math.sin(self.time + i) * 0.1
            
            color = self.colors['accent_primary']
            overlay = frame.copy()
            cv2.circle(overlay, (x, y), size, color, -1)
            frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    def render_header(self, frame: np.ndarray, w: int, title: str, subtitle: str):
        """Elegant header with sophisticated typography"""
        # Title
        title_size = 1.6
        title_y = 70
        title_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_size, 3)[0][0]
        title_x = (w - title_width) // 2
        
        # Glow effect
        glow = frame.copy()
        cv2.putText(glow, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, title_size, 
                   self.colors['glow_gold'], 20, cv2.LINE_AA)
        frame = cv2.addWeighted(frame, 0.92, glow, 0.08, 0)
        
        # Main title
        cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, title_size, 
                   self.colors['accent_gold'], 3, cv2.LINE_AA)
        
        # Subtitle
        subtitle_size = 0.65
        subtitle_width = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, subtitle_size, 1)[0][0]
        subtitle_x = (w - subtitle_width) // 2
        cv2.putText(frame, subtitle, (subtitle_x, 100), cv2.FONT_HERSHEY_SIMPLEX, subtitle_size,
                   self.colors['text_secondary'], 1, cv2.LINE_AA)
        
        # Decorative line
        line_y = 115
        line_padding = 200
        cv2.line(frame, (line_padding, line_y), (w - line_padding, line_y), 
                self.colors['border_subtle'], 1, cv2.LINE_AA)
    
    def render_stats_panel_elegant(self, frame: np.ndarray, x: int, y: int, width: int):
        """Elegant stats with smooth animations"""
        panel_height = 420
        
        # Glass morphism card
        self.draw_glass_card(frame, x, y, width, panel_height)
        
        # Header
        cv2.putText(frame, 'Financial Position', (x + 20, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, self.colors['text_primary'], 2, cv2.LINE_AA)
        
        # Animated stats
        stats_config = [
            ('Stability', 'stability', self.colors['accent_green']),
            ('Risk Exposure', 'risk', self.colors['accent_red']),
            ('Liquidity', 'liquidity', self.colors['accent_primary']),
            ('Credit Rating', 'credit_score', self.colors['accent_gold'])
        ]
        
        stat_y = y + 75
        for label, key, color in stats_config:
            value = self.animated_stats[key].get()
            self.render_stat_elegant(frame, x + 20, stat_y, width - 40, label, value, color)
            stat_y += 85
    
    def render_stat_elegant(self, frame: np.ndarray, x: int, y: int, width: int, 
                           label: str, value: float, color: Tuple[int, int, int]):
        """Individual stat with smooth bar animation"""
        # Label and value
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 
                   self.colors['text_secondary'], 1, cv2.LINE_AA)
        
        value_text = f'{int(value)}'
        value_width = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.putText(frame, value_text, (x + width - value_width, y + 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        # Animated bar
        bar_y = y + 15
        bar_height = 8
        
        # Background track
        cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + bar_height), 
                     self.colors['bg_secondary'], -1)
        
        # Filled portion with glow
        fill_width = int((value / 100.0) * width)
        if fill_width > 2:
            # Glow
            glow = frame.copy()
            cv2.rectangle(glow, (x, bar_y - 2), (x + fill_width, bar_y + bar_height + 2), 
                         color, -1)
            frame = cv2.addWeighted(frame, 0.9, glow, 0.1, 0)
            
            # Main bar with gradient
            overlay = frame.copy()
            for i in range(fill_width):
                alpha = 0.7 + 0.3 * (i / fill_width)
                bar_color = tuple(int(c * alpha) for c in color)
                cv2.line(overlay, (x + i, bar_y), (x + i, bar_y + bar_height), bar_color, 1)
            frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Percentage indicator
        if fill_width > 0:
            indicator_x = x + fill_width
            cv2.circle(frame, (indicator_x, bar_y + bar_height // 2), 4, color, -1)
            cv2.circle(frame, (indicator_x, bar_y + bar_height // 2), 6, color, 1)
    
    def render_context_elegant(self, frame: np.ndarray, x: int, y: int, width: int, text: str):
        """Elegant context panel"""
        # Calculate height needed
        words = text.split()
        lines = []
        current_line = ""
        max_width = 45
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_width:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        
        height = 30 + len(lines) * 28 + 20
        
        # Glass card
        self.draw_glass_card(frame, x, y, width, height)
        
        # Text
        text_y = y + 28
        for line in lines:
            cv2.putText(frame, line, (x + 20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       self.colors['text_secondary'], 1, cv2.LINE_AA)
            text_y += 28
    
    def render_image_elegant(self, frame: np.ndarray, x: int, y: int, image_key: str):
        """Image with elegant frame"""
        img = self.images[image_key]
        
        # Glow effect
        glow = frame.copy()
        glow_size = 8
        cv2.rectangle(glow, (x - glow_size, y - glow_size), 
                     (x + 320 + glow_size, y + 320 + glow_size),
                     self.colors['glow_blue'], -1)
        frame = cv2.addWeighted(frame, 0.95, glow, 0.05, 0)
        
        # Frame
        cv2.rectangle(frame, (x - 3, y - 3), (x + 323, y + 323), 
                     self.colors['border_bright'], 2, cv2.LINE_AA)
        
        # Image
        frame = self.alpha_blend(frame, img, x, y)
    
    def render_preview_elegant(self, frame: np.ndarray, x: int, y: int, width: int, decision: Decision):
        """Elegant preview panel"""
        height = 300
        
        self.draw_glass_card(frame, x, y, width, height)
        
        # Header
        cv2.putText(frame, 'Impact Analysis', (x + 20, y + 35), cv2.FONT_HERSHEY_SIMPLEX,
                   0.75, self.colors['accent_gold'], 2, cv2.LINE_AA)
        
        # Impacts
        impact_y = y + 75
        for stat, change in decision.impact.items():
            if change == 0:
                continue
            
            label = stat.replace('_', ' ').title()
            sign = '+' if change > 0 else ''
            color = self.colors['accent_green'] if change > 0 else self.colors['accent_red']
            arrow = '↑' if change > 0 else '↓'
            
            cv2.putText(frame, f"{arrow} {label}", (x + 25, impact_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors['text_secondary'], 1, cv2.LINE_AA)
            
            change_text = f'{sign}{int(change)}'
            cv2.putText(frame, change_text, (x + width - 60, impact_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            
            impact_y += 45
    
    def render_phase_info(self, frame: np.ndarray, x: int, y: int, width: int):
        """Phase information when no hover"""
        height = 200
        self.draw_glass_card(frame, x, y, width, height)
        
        cv2.putText(frame, 'Guidance', (x + 20, y + 35), cv2.FONT_HERSHEY_SIMPLEX,
                   0.75, self.colors['text_primary'], 2, cv2.LINE_AA)
        
        tips = [
            'Hover over decisions',
            'to preview impact',
            '',
            'Hold for 1 second',
            'to confirm choice'
        ]
        
        tip_y = y + 70
        for tip in tips:
            cv2.putText(frame, tip, (x + 25, tip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                       self.colors['text_tertiary'], 1, cv2.LINE_AA)
            tip_y += 26
    
    def render_decision_card(self, frame: np.ndarray, x: int, y: int, w: int, h: int,
                            decision: Decision, idx: int, is_hovered: bool, hand_pos: Optional[Tuple[int, int]]):
        """Elegant decision card with smooth hover animation"""
        # Update hover animation
        if is_hovered:
            self.button_hover_alpha[idx].set_target(1.0)
        else:
            self.button_hover_alpha[idx].set_target(0.0)
        
        hover_amount = self.button_hover_alpha[idx].get()
        
        # Lift effect
        lift = int(hover_amount * 8)
        y_adjusted = y - lift
        
        # Glow
        if hover_amount > 0.01:
            glow = frame.copy()
            glow_expand = int(6 + hover_amount * 8)
            cv2.rectangle(glow, (x - glow_expand, y_adjusted - glow_expand),
                         (x + w + glow_expand, y_adjusted + h + glow_expand),
                         self.colors['glow_blue'], -1)
            frame = cv2.addWeighted(frame, 1 - hover_amount * 0.15, glow, hover_amount * 0.15, 0)
        
        # Card background
        card_color = tuple(int(c + hover_amount * 15) for c in self.colors['bg_card'])
        cv2.rectangle(frame, (x, y_adjusted), (x + w, y_adjusted + h), card_color, -1)
        
        # Border
        border_color = self.colors['accent_primary'] if hover_amount > 0.5 else self.colors['border_bright']
        border_thickness = 2 if hover_amount > 0.5 else 1
        cv2.rectangle(frame, (x, y_adjusted), (x + w, y_adjusted + h), border_color, border_thickness, cv2.LINE_AA)
        
        # Title
        cv2.putText(frame, decision.title, (x + 20, y_adjusted + 40), cv2.FONT_HERSHEY_SIMPLEX,
                   0.65, self.colors['text_primary'], 2, cv2.LINE_AA)
        
        # Description
        desc_lines = self.wrap_text(decision.description, 28)
        desc_y = y_adjusted + 70
        for line in desc_lines:
            cv2.putText(frame, line, (x + 20, desc_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                       self.colors['text_tertiary'], 1, cv2.LINE_AA)
            desc_y += 22
        
        # Progress bar
        if is_hovered and hand_pos:
            progress = min(1.0, (time.time() - self.hover_start_time) / self.hover_duration)
            
            bar_y = y_adjusted + h - 25
            bar_width = w - 40
            
            # Track
            cv2.rectangle(frame, (x + 20, bar_y), (x + 20 + bar_width, bar_y + 6),
                         self.colors['bg_secondary'], -1)
            
            # Progress with gradient
            fill_width = int(progress * bar_width)
            if fill_width > 0:
                for i in range(fill_width):
                    alpha = 0.6 + 0.4 * (i / bar_width)
                    color = tuple(int(c * alpha) for c in self.colors['accent_green'])
                    cv2.line(frame, (x + 20 + i, bar_y), (x + 20 + i, bar_y + 6), color, 1)
                
                # Indicator
                cv2.circle(frame, (x + 20 + fill_width, bar_y + 3), 5, self.colors['accent_green'], -1)
                cv2.circle(frame, (x + 20 + fill_width, bar_y + 3), 7, self.colors['accent_green'], 1)
            
            # Percentage
            percent_text = f'{int(progress * 100)}%'
            cv2.putText(frame, percent_text, (x + w - 60, bar_y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, self.colors['accent_green'], 1, cv2.LINE_AA)
    
    def render_fact_screen(self, frame: np.ndarray) -> np.ndarray:
        """Elegant fact presentation"""
        h, w = frame.shape[:2]
        
        # Animated fade
        fade_progress = min(1.0, (time.time() - self.fact_display_time) / 0.5)
        alpha = min(0.85, fade_progress * 0.85)
        
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:] = self.colors['bg_primary']
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        # Content panel
        panel_width = 900
        panel_height = 300
        panel_x = (w - panel_width) // 2
        panel_y = (h - panel_height) // 2
        
        self.draw_glass_card(frame, panel_x, panel_y, panel_width, panel_height)
        
        # Icon
        icon_size = 40
        icon_x = panel_x + panel_width // 2
        icon_y = panel_y + 60
        cv2.circle(frame, (icon_x, icon_y), icon_size, self.colors['accent_gold'], 2)
        cv2.putText(frame, 'i', (icon_x - 8, icon_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                   self.colors['accent_gold'], 2, cv2.LINE_AA)
        
        # Title
        cv2.putText(frame, 'Key Insight', (panel_x + panel_width//2 - 80, panel_y + 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['text_primary'], 2, cv2.LINE_AA)
        
        # Fact text
        fact_lines = self.wrap_text(self.current_fact, 70)
        fact_y = panel_y + 175
        for line in fact_lines:
            line_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            cv2.putText(frame, line, (panel_x + (panel_width - line_width)//2, fact_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors['text_secondary'], 1, cv2.LINE_AA)
            fact_y += 32
        
        if time.time() - self.fact_display_time > self.fact_duration:
            self.showing_fact = False
            self.phase += 1
            if self.phase >= len(self.phases):
                self.completed = True
        
        return frame
    
    def render_reflection(self, frame: np.ndarray) -> np.ndarray:
        """Sophisticated completion screen"""
        h, w = frame.shape[:2]
        
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:] = self.colors['bg_primary']
        frame = cv2.addWeighted(frame, 0.15, overlay, 0.85, 0)
        
        # Header
        cv2.putText(frame, 'Financial Simulation Complete', (w//2 - 320, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.colors['accent_gold'], 3, cv2.LINE_AA)
        
        # Stats summary
        self.render_stats_panel_elegant(frame, 60, 130, 350)
        
        # Analysis
        analysis_x = 470
        analysis_y = 130
        analysis_width = 750
        analysis_height = 450
        
        self.draw_glass_card(frame, analysis_x, analysis_y, analysis_width, analysis_height)
        
        cv2.putText(frame, 'Strategy Assessment', (analysis_x + 25, analysis_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.95, self.colors['text_primary'], 2, cv2.LINE_AA)
        
        analysis = self.generate_analysis()
        text_y = analysis_y + 90
        for line in analysis:
            cv2.putText(frame, line, (analysis_x + 30, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                       self.colors['text_secondary'], 1, cv2.LINE_AA)
            text_y += 32
        
        # Achievement badge
        if (self.stats.stability >= 70 and self.stats.liquidity >= 30 and self.stats.risk < 60):
            badge_y = analysis_y + analysis_height - 70
            
            # Badge background
            badge_x = analysis_x + analysis_width // 2
            cv2.circle(frame, (badge_x, badge_y), 45, self.colors['accent_gold'], 3)
            cv2.circle(frame, (badge_x, badge_y), 35, self.colors['accent_gold'], 1)
            
            cv2.putText(frame, 'Balanced Strategy Achieved', (analysis_x + 180, badge_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.colors['accent_gold'], 2, cv2.LINE_AA)
        
        # Restart button
        button_width = 280
        button_height = 65
        button_x = w // 2 - button_width // 2
        button_y = h - 120
        
        self.draw_glass_card(frame, button_x, button_y, button_width, button_height)
        cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + button_height),
                     self.colors['accent_primary'], 2, cv2.LINE_AA)
        
        cv2.putText(frame, 'Restart Simulation', (button_x + 45, button_y + 43),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text_primary'], 2, cv2.LINE_AA)
        cv2.putText(frame, 'Press R', (button_x + 105, button_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['text_tertiary'], 1, cv2.LINE_AA)
        
        return frame
    
    def generate_analysis(self) -> List[str]:
        analysis = []
        
        if self.stats.risk > 70:
            analysis.append("High-Risk Profile: Your strategy prioritized short-term gains over")
            analysis.append("long-term stability. This approach increases vulnerability to market volatility.")
        elif self.stats.risk < 40:
            analysis.append("Conservative Approach: Risk mitigation was central to your strategy.")
            analysis.append("This provides resilience but may limit growth potential.")
        else:
            analysis.append("Balanced Risk Management: Your strategy achieved equilibrium between")
            analysis.append("security and opportunity, optimizing for sustainable wealth building.")
        
        analysis.append("")
        
        if self.stats.liquidity > 60:
            analysis.append("Strong Liquidity Position: Excellent cash reserves provide optionality")
            analysis.append("and protection against unexpected financial shocks.")
        elif self.stats.liquidity < 30:
            analysis.append("Liquidity Constraints: Limited cash reserves increase vulnerability")
            analysis.append("to financial disruption and reduce strategic flexibility.")
        
        analysis.append("")
        
        if self.insurance_level >= 1:
            analysis.append("Risk Transfer Implemented: Insurance coverage demonstrates mature")
            analysis.append("financial planning and protection against catastrophic losses.")
        
        return analysis
    
    def draw_glass_card(self, frame: np.ndarray, x: int, y: int, width: int, height: int):
        """Glass morphism card effect"""
        # Main card
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), self.colors['bg_card'], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['border_subtle'], 1, cv2.LINE_AA)
        
        # Top highlight
        highlight = frame.copy()
        cv2.line(highlight, (x + 1, y + 1), (x + width - 1, y + 1), 
                (255, 255, 255), 1, cv2.LINE_AA)
        frame = cv2.addWeighted(frame, 0.95, highlight, 0.05, 0)
    
    def check_hover(self, hand_pos: Optional[Tuple[int, int]], x: int, y: int, w: int, h: int) -> bool:
        if hand_pos is None:
            return False
        hx, hy = hand_pos
        return x <= hx <= x + w and y <= hy <= y + h
    
    def update_hover(self, hand_pos: Optional[Tuple[int, int]]):
        if self.phase >= len(self.phases) or self.completed or self.showing_fact:
            return
        
        decisions = self.phases[self.phase]['decisions']
        if not decisions:
            return
        
        h, w = 720, 1280
        button_y = h - 180
        card_width = 320
        card_spacing = 40
        total_width = len(decisions) * card_width + (len(decisions) - 1) * card_spacing
        start_x = (w - total_width) // 2
        
        current_hover = -1
        for i in range(len(decisions)):
            card_x = start_x + i * (card_width + card_spacing)
            if self.check_hover(hand_pos, card_x, button_y, card_width, 150):
                current_hover = i
                break
        
        if current_hover != self.hovered_button:
            self.hovered_button = current_hover
            self.hover_start_time = time.time()
        
        if current_hover >= 0:
            hover_time = time.time() - self.hover_start_time
            if hover_time >= self.hover_duration:
                self.apply_decision(current_hover)
                self.hovered_button = -1
    
    def apply_decision(self, decision_idx: int):
        if self.phase >= len(self.phases):
            return
        
        phase_data = self.phases[self.phase]
        decision = phase_data['decisions'][decision_idx]
        
        self.decision_history.append(decision.title)
        
        if self.phase == 1:
            if 'Comprehensive' in decision.title:
                self.insurance_level = 2
            elif 'Essential' in decision.title:
                self.insurance_level = 1
        
        if self.phase == 4:
            self.calculate_investment_outcome(decision_idx)
        
        for stat, change in decision.impact.items():
            current_value = getattr(self.target_stats, stat)
            setattr(self.target_stats, stat, current_value + change)
        
        self.target_stats.knowledge += 15
        self.target_stats.clamp()
        
        self.current_fact = phase_data['fact']
        self.showing_fact = True
        self.fact_display_time = time.time()
    
    def alpha_blend(self, background: np.ndarray, foreground: np.ndarray, x: int, y: int) -> np.ndarray:
        bg_h, bg_w = background.shape[:2]
        fg_h, fg_w = foreground.shape[:2]
        
        if x < 0 or y < 0 or x + fg_w > bg_w or y + fg_h > bg_h:
            return background
        
        if foreground.shape[2] == 4:
            alpha = foreground[:, :, 3] / 255.0
            foreground_rgb = foreground[:, :, :3]
        else:
            alpha = np.ones((fg_h, fg_w))
            foreground_rgb = foreground
        
        roi = background[y:y+fg_h, x:x+fg_w]
        for c in range(3):
            roi[:, :, c] = (alpha * foreground_rgb[:, :, c] + (1 - alpha) * roi[:, :, c])
        
        return background
    
    def wrap_text(self, text: str, max_chars: int) -> List[str]:
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            lines.append(current_line.strip())
        
        return lines
    
    def evaluate(self) -> Dict:
        return {
            'stability': self.stats.stability,
            'risk': self.stats.risk,
            'liquidity': self.stats.liquidity,
            'credit_score': self.stats.credit_score,
            'knowledge': self.stats.knowledge,
            'decisions': self.decision_history,
            'balanced_strategy': (self.stats.stability >= 70 and self.stats.liquidity >= 30 and self.stats.risk < 60)
        }
    
    def reset(self):
        self.stats = FinancialStats()
        self.target_stats = FinancialStats()
        
        for key in self.animated_stats:
            self.animated_stats[key] = AnimatedValue(50.0 if key == 'stability' else 
                                                     30.0 if key == 'risk' else
                                                     40.0 if key == 'liquidity' else
                                                     50.0 if key == 'credit_score' else 0.0, 0.06)
        
        self.phase = 0
        self.active = False
        self.completed = False
        self.hovered_button = -1
        self.hover_start_time = 0
        self.showing_fact = False
        self.decision_history = []
        self.insurance_level = 0
        self.crisis_outcome = ""
        self.time = 0.0
        self.phases[3]['decisions'] = []
        self.phases[3]['context'] = None

_finance_sim = None

def run_finance(frame: np.ndarray, hand_pos: Optional[Tuple[int, int]]) -> np.ndarray:
    global _finance_sim
    
    if _finance_sim is None:
        _finance_sim = FinanceSimulation()
        _finance_sim.active = True
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r') and _finance_sim.completed:
        _finance_sim.reset()
        _finance_sim.active = True
    
    _finance_sim.update_hover(hand_pos)
    
    return _finance_sim.run_finance(frame, hand_pos)

def evaluate() -> Dict:
    global _finance_sim
    if _finance_sim:
        return _finance_sim.evaluate()
    return {}

def reset():
    global _finance_sim
    if _finance_sim:
        _finance_sim.reset()

if __name__ == "__main__":
    print("Finance Simulation - Professional Edition")
    print("Sophisticated animations and refined aesthetics")