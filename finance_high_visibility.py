"""
Advanced Interactive Finance Simulation Module
Professional financial literacy training simulator with AR webcam integration
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time

@dataclass
class FinancialStats:
    """Financial metrics with interdependencies"""
    stability: float = 50.0
    risk: float = 30.0
    liquidity: float = 40.0
    credit_score: float = 50.0
    knowledge: float = 0.0
    
    def clamp(self):
        """Ensure all stats stay within 0-100 range"""
        self.stability = max(0, min(100, self.stability))
        self.risk = max(0, min(100, self.risk))
        self.liquidity = max(0, min(100, self.liquidity))
        self.credit_score = max(0, min(100, self.credit_score))
        self.knowledge = max(0, min(100, self.knowledge))

@dataclass
class Decision:
    """Represents a financial decision option"""
    title: str
    impact: Dict[str, float]
    description: str

class FinanceSimulation:
    def __init__(self):
        # Core state
        self.stats = FinancialStats()
        self.target_stats = FinancialStats()
        self.phase = 0
        self.active = False
        self.completed = False
        
        # UI state
        self.hovered_button = -1
        self.hover_start_time = 0
        self.hover_duration = 1.0  # 1 second to select
        self.showing_fact = False
        self.fact_display_time = 0
        self.fact_duration = 3.0
        self.current_fact = ""
        self.transition_alpha = 0.0
        self.transitioning = False
        
        # Decision history
        self.decision_history = []
        self.insurance_level = 0  # 0=none, 1=basic, 2=comprehensive
        self.crisis_outcome = ""
        
        # Animation
        self.stat_lerp_speed = 0.05
        self.glow_pulse = 0.0
        
        # Images
        self.images = {}
        self.load_images()
        
        # Phase definitions
        self.phases = self.define_phases()
        
    def load_images(self):
        """Load scenario images with safe fallback"""
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
                    # Resize to standard size
                    img = cv2.resize(img, (350, 350))
                    self.images[key] = img
                else:
                    # Create placeholder
                    self.images[key] = self.create_placeholder(key)
            except:
                self.images[key] = self.create_placeholder(key)
    
    def create_placeholder(self, label: str) -> np.ndarray:
        """Create placeholder image if asset missing"""
        img = np.zeros((350, 350, 4), dtype=np.uint8)
        img[:, :, 3] = 200  # Semi-transparent
        img[:, :, 0:3] = (40, 60, 80)  # Dark blue-grey
        
        # Add text
        cv2.putText(img, label.upper(), (50, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 180, 255), 2)
        return img
    
    def define_phases(self) -> List[Dict]:
        """Define all simulation phases"""
        return [
            {
                'title': 'Housing Commitment',
                'context': 'Choose your living arrangement. This will significantly\nimpact your monthly expenses and financial flexibility.',
                'image': 'housing',
                'decisions': [
                    Decision(
                        'Luxury Apartment',
                        {'stability': -15, 'risk': 15, 'liquidity': -25, 'credit_score': 10},
                        'High monthly rent, but impressive address'
                    ),
                    Decision(
                        'Shared Apartment',
                        {'stability': 10, 'risk': -5, 'liquidity': 10, 'credit_score': 5},
                        'Affordable with roommates, balanced approach'
                    ),
                    Decision(
                        'Stay With Parents',
                        {'stability': 15, 'risk': -10, 'liquidity': 25, 'credit_score': 0},
                        'No rent, maximize savings potential'
                    )
                ],
                'fact': 'Financial experts recommend housing costs stay below 30% of income.'
            },
            {
                'title': 'Insurance Planning',
                'context': 'Protect yourself against unexpected events. Insurance is\nan investment in peace of mind and financial security.',
                'image': 'insurance',
                'decisions': [
                    Decision(
                        'Comprehensive Insurance',
                        {'stability': 20, 'risk': -20, 'liquidity': -15, 'credit_score': 10},
                        'Full coverage: health, life, property'
                    ),
                    Decision(
                        'Basic Coverage',
                        {'stability': 10, 'risk': -5, 'liquidity': -5, 'credit_score': 5},
                        'Essential protection only'
                    ),
                    Decision(
                        'No Insurance',
                        {'stability': -10, 'risk': 25, 'liquidity': 5, 'credit_score': -5},
                        'Save money, accept all risk'
                    )
                ],
                'fact': 'Medical emergencies are one of the leading causes of personal debt globally.'
            },
            {
                'title': 'Lifestyle & Debt',
                'context': 'Your spending habits will shape your financial future.\nCredit can be a tool or a trap.',
                'image': 'credit_card',
                'decisions': [
                    Decision(
                        'Credit Card EMI Lifestyle',
                        {'stability': -20, 'risk': 30, 'liquidity': 10, 'credit_score': -15},
                        'Live now, pay later with interest'
                    ),
                    Decision(
                        'Controlled Spending',
                        {'stability': 15, 'risk': -10, 'liquidity': 5, 'credit_score': 10},
                        'Budget-conscious, occasional treats'
                    ),
                    Decision(
                        'Aggressive Saving',
                        {'stability': 25, 'risk': -15, 'liquidity': 20, 'credit_score': 5},
                        'Minimize expenses, maximize savings'
                    )
                ],
                'fact': 'High-interest debt compounds quickly if unpaid. Credit card APR can exceed 30%.'
            },
            {
                'title': 'Crisis Event',
                'context': None,  # Dynamic based on crisis
                'image': 'crisis',
                'decisions': [],  # Generated dynamically
                'fact': 'An emergency fund should cover at least 3-6 months of expenses.'
            },
            {
                'title': 'Investment Opportunity',
                'context': 'A chance to grow your wealth. Higher returns often\nmean higher risk. Choose wisely based on your position.',
                'image': 'investment',
                'decisions': [
                    Decision(
                        'Startup Investment',
                        {'stability': 0, 'risk': 25, 'liquidity': -20, 'credit_score': 0},
                        'High risk, high potential return'
                    ),
                    Decision(
                        'Fixed Deposit',
                        {'stability': 15, 'risk': -10, 'liquidity': -10, 'credit_score': 10},
                        'Safe, steady returns guaranteed'
                    ),
                    Decision(
                        'No Investment',
                        {'stability': -5, 'risk': 0, 'liquidity': 5, 'credit_score': 0},
                        'Keep cash liquid, no growth'
                    )
                ],
                'fact': 'Diversification reduces long-term investment risk. Never invest what you cannot afford to lose.'
            }
        ]
    
    def generate_crisis(self) -> Dict:
        """Generate crisis event based on current stats"""
        # Determine crisis type based on decisions
        import random
        
        crises = [
            {
                'type': 'Medical Emergency',
                'description': 'Unexpected hospitalization required. Immediate payment needed.',
            },
            {
                'type': 'Job Loss',
                'description': 'Company downsizing. Income stopped for 3 months.',
            },
            {
                'type': 'Vehicle Breakdown',
                'description': 'Critical repairs needed. No vehicle, no commute.',
            }
        ]
        
        crisis = random.choice(crises)
        
        # Calculate impact based on preparation
        base_stability_loss = 25
        base_liquidity_loss = 30
        base_risk_increase = 20
        
        # Insurance mitigation
        if self.insurance_level == 2:  # Comprehensive
            base_stability_loss *= 0.3
            base_liquidity_loss *= 0.4
            base_risk_increase *= 0.2
            crisis['outcome'] = 'Insurance covered most costs. Minimal impact.'
        elif self.insurance_level == 1:  # Basic
            base_stability_loss *= 0.6
            base_liquidity_loss *= 0.7
            base_risk_increase *= 0.5
            crisis['outcome'] = 'Insurance helped, but significant out-of-pocket expenses remain.'
        else:  # No insurance
            crisis['outcome'] = 'Full financial impact absorbed. Severe setback.'
        
        # Liquidity mitigation
        if self.stats.liquidity > 60:
            base_stability_loss *= 0.7
            crisis['outcome'] += ' Emergency fund cushioned the blow.'
        elif self.stats.liquidity < 30:
            base_stability_loss *= 1.3
            base_risk_increase *= 1.2
            crisis['outcome'] += ' Low liquidity made this crisis worse.'
        
        # Credit score impact
        if self.stats.credit_score < 40:
            base_risk_increase *= 1.3
            crisis['outcome'] += ' Poor credit limited recovery options.'
        
        crisis['impact'] = {
            'stability': -base_stability_loss,
            'risk': base_risk_increase,
            'liquidity': -base_liquidity_loss,
            'credit_score': -10
        }
        
        self.crisis_outcome = crisis['outcome']
        return crisis
    
    def calculate_investment_outcome(self, decision_idx: int):
        """Modify investment outcome based on financial position"""
        decision = self.phases[4]['decisions'][decision_idx]
        
        if decision_idx == 0:  # Startup investment
            # Success chance based on stats
            success_threshold = 40 + (self.stats.stability * 0.3) + (self.stats.credit_score * 0.2)
            success_threshold -= (self.stats.risk * 0.3)
            
            import random
            if random.random() * 100 < success_threshold:
                # Success
                decision.impact = {
                    'stability': 20,
                    'risk': -10,
                    'liquidity': 30,
                    'credit_score': 15
                }
                self.current_fact = 'Investment succeeded! Diversification reduces long-term risk.'
            else:
                # Failure
                decision.impact = {
                    'stability': -25,
                    'risk': 15,
                    'liquidity': -20,
                    'credit_score': -10
                }
                self.current_fact = 'Investment failed. Never invest what you cannot afford to lose.'
    
    def run_finance(self, frame: np.ndarray, hand_pos: Optional[Tuple[int, int]]) -> np.ndarray:
        """Main render loop"""
        if not self.active:
            return frame
        
        # Update animations
        self.update_animations()
        
        # Smooth stat transitions
        self.lerp_stats()
        
        # Render based on state
        if self.showing_fact:
            frame = self.render_fact_screen(frame)
        elif self.completed:
            frame = self.render_reflection(frame)
        else:
            frame = self.render_phase(frame, hand_pos)
        
        # Visual effects based on stats
        frame = self.apply_stat_effects(frame)
        
        return frame
    
    def update_animations(self):
        """Update animation timers"""
        self.glow_pulse = (self.glow_pulse + 0.05) % (2 * np.pi)
        
        if self.transitioning:
            self.transition_alpha += 0.1
            if self.transition_alpha >= 1.0:
                self.transitioning = False
                self.transition_alpha = 0.0
    
    def lerp_stats(self):
        """Smooth stat bar animations"""
        self.stats.stability += (self.target_stats.stability - self.stats.stability) * self.stat_lerp_speed
        self.stats.risk += (self.target_stats.risk - self.stats.risk) * self.stat_lerp_speed
        self.stats.liquidity += (self.target_stats.liquidity - self.stats.liquidity) * self.stat_lerp_speed
        self.stats.credit_score += (self.target_stats.credit_score - self.stats.credit_score) * self.stat_lerp_speed
        self.stats.knowledge += (self.target_stats.knowledge - self.stats.knowledge) * self.stat_lerp_speed
    
    def render_phase(self, frame: np.ndarray, hand_pos: Optional[Tuple[int, int]]) -> np.ndarray:
        """Render current phase UI"""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Generate crisis dynamically
        if self.phase == 3 and not self.phases[3]['decisions']:
            crisis = self.generate_crisis()
            self.phases[3]['context'] = crisis['description']
            self.phases[3]['decisions'] = [
                Decision('Accept Impact', crisis['impact'], crisis['outcome'])
            ]
        
        phase_data = self.phases[self.phase]
        
        # Title
        title_y = 60
        cv2.putText(frame, phase_data['title'], (w//2 - 200, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 180, 255), 3)
        
        # Stats panel (left side)
        self.render_stats_panel(frame, 50, 120)
        
        # Context panel (center)
        if phase_data['context']:
            self.render_context_panel(frame, w//2 - 250, 120, phase_data['context'])
        
        # Image panel (right side)
        self.render_image_panel(frame, w - 420, 120, phase_data['image'])
        
        # Decision buttons
        button_y = h - 300
        decisions = phase_data['decisions']
        
        if len(decisions) > 0:
            button_width = 280
            button_spacing = 40
            total_width = len(decisions) * button_width + (len(decisions) - 1) * button_spacing
            start_x = (w - total_width) // 2
            
            for i, decision in enumerate(decisions):
                button_x = start_x + i * (button_width + button_spacing)
                is_hovered = self.check_hover(hand_pos, button_x, button_y, button_width, 120)
                
                self.render_decision_button(frame, button_x, button_y, button_width, 120, 
                                            decision, i, is_hovered, hand_pos)
        
        # Preview panel (if hovering)
        if self.hovered_button >= 0 and self.hovered_button < len(decisions):
            self.render_preview_panel(frame, w//2 - 200, button_y - 180, 
                                     decisions[self.hovered_button])
        
        return frame
    
    def render_stats_panel(self, frame: np.ndarray, x: int, y: int):
        """Render stats with animated bars"""
        panel_width = 320
        panel_height = 400
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                     (40, 50, 60), -1)
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                     (100, 180, 255), 2)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Title
        cv2.putText(frame, 'FINANCIAL STATUS', (x + 20, y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 180, 255), 2)
        
        # Stats
        stats_list = [
            ('Stability', self.stats.stability, (80, 200, 120)),
            ('Risk', self.stats.risk, (100, 100, 255)),
            ('Liquidity', self.stats.liquidity, (150, 150, 255)),
            ('Credit Score', self.stats.credit_score, (255, 200, 100))
        ]
        
        bar_y = y + 70
        for label, value, color in stats_list:
            self.render_stat_bar(frame, x + 20, bar_y, panel_width - 40, 
                               label, value, color)
            bar_y += 75
        
        # Glow effect for high stability
        if self.stats.stability > 70:
            glow_alpha = 0.3 + 0.2 * np.sin(self.glow_pulse)
            glow = frame.copy()
            cv2.rectangle(glow, (x-3, y-3), (x + panel_width+3, y + panel_height+3),
                         (80, 255, 120), 6)
            frame = cv2.addWeighted(frame, 1-glow_alpha, glow, glow_alpha, 0)
    
    def render_stat_bar(self, frame: np.ndarray, x: int, y: int, width: int,
                       label: str, value: float, color: Tuple[int, int, int]):
        """Render single animated stat bar"""
        # Label
        cv2.putText(frame, label, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Value
        cv2.putText(frame, f'{int(value)}', (x + width - 40, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Bar background
        bar_y = y + 10
        cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + 20),
                     (30, 30, 30), -1)
        cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + 20),
                     (100, 100, 100), 1)
        
        # Bar fill
        fill_width = int((value / 100.0) * width)
        if fill_width > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, bar_y), (x + fill_width, bar_y + 20),
                         color, -1)
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    
    def render_context_panel(self, frame: np.ndarray, x: int, y: int, text: str):
        """Render context description"""
        panel_width = 500
        panel_height = 150
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                     (30, 40, 50), -1)
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                     (100, 180, 255), 2)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # Multi-line text
        lines = text.split('\n')
        text_y = y + 40
        for line in lines:
            cv2.putText(frame, line, (x + 20, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            text_y += 30
    
    def render_image_panel(self, frame: np.ndarray, x: int, y: int, image_key: str):
        """Render scenario image with glow"""
        if image_key not in self.images:
            return
        
        img = self.images[image_key]
        
        # Glow behind image
        glow_alpha = 0.2 + 0.1 * np.sin(self.glow_pulse)
        glow = frame.copy()
        cv2.rectangle(glow, (x-5, y-5), (x + 360, y + 360),
                     (100, 180, 255), 10)
        frame = cv2.addWeighted(frame, 1-glow_alpha, glow, glow_alpha, 0)
        
        # Draw image with alpha blending
        frame = self.alpha_blend(frame, img, x, y)
        
        return frame
    
    def render_decision_button(self, frame: np.ndarray, x: int, y: int, 
                               w: int, h: int, decision: Decision, 
                               idx: int, is_hovered: bool, hand_pos: Optional[Tuple[int, int]]):
        """Render interactive decision button"""
        # Hover effect
        color = (100, 180, 255) if is_hovered else (80, 120, 160)
        thickness = 3 if is_hovered else 2
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 40, 50), -1)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
        alpha = 0.5 if is_hovered else 0.7
        frame = cv2.addWeighted(frame, alpha, overlay, 1-alpha, 0)
        
        # Title
        title_lines = self.wrap_text(decision.title, 20)
        text_y = y + 30
        for line in title_lines:
            cv2.putText(frame, line, (x + 15, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)
            text_y += 30
        
        # Description
        desc_lines = self.wrap_text(decision.description, 30)
        text_y += 10
        for line in desc_lines:
            cv2.putText(frame, line, (x + 15, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            text_y += 20
        
        # Hover progress
        if is_hovered and hand_pos:
            progress = min(1.0, (time.time() - self.hover_start_time) / self.hover_duration)
            
            # Progress bar
            bar_width = int(progress * (w - 20))
            cv2.rectangle(frame, (x + 10, y + h - 15), 
                         (x + 10 + bar_width, y + h - 5),
                         (100, 255, 180), -1)
            
            # Glow when complete
            if progress >= 1.0:
                glow = frame.copy()
                cv2.rectangle(glow, (x-3, y-3), (x + w + 3, y + h + 3),
                             (100, 255, 180), 6)
                frame = cv2.addWeighted(frame, 0.6, glow, 0.4, 0)
    
    def render_preview_panel(self, frame: np.ndarray, x: int, y: int, decision: Decision):
        """Render impact preview before selection"""
        panel_width = 400
        panel_height = 160
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                     (20, 30, 40), -1)
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                     (255, 200, 100), 3)
        frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        
        # Title
        cv2.putText(frame, 'PROJECTED IMPACT', (x + 20, y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 150), 2)
        
        # Impact list
        text_y = y + 60
        for stat, change in decision.impact.items():
            if change == 0:
                continue
            
            label = stat.replace('_', ' ').title()
            sign = '+' if change > 0 else ''
            color = (100, 255, 100) if change > 0 else (100, 100, 255)
            arrow = '↑' if change > 0 else '↓'
            
            text = f"{arrow} {label}: {sign}{int(change)}"
            cv2.putText(frame, text, (x + 30, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            text_y += 25
        
        return frame
    
    def render_fact_screen(self, frame: np.ndarray) -> np.ndarray:
        """Display educational fact after decision"""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)
        
        # Fact panel
        panel_width = 800
        panel_height = 300
        panel_x = (w - panel_width) // 2
        panel_y = (h - panel_height) // 2
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (30, 50, 70), -1)
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (100, 255, 180), 4)
        frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)
        
        # Title
        cv2.putText(frame, 'FINANCIAL INSIGHT', (panel_x + 40, panel_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 180), 3)
        
        # Fact text
        fact_lines = self.wrap_text(self.current_fact, 70)
        text_y = panel_y + 120
        for line in fact_lines:
            cv2.putText(frame, line, (panel_x + 40, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 240), 2)
            text_y += 40
        
        # Check if fact display time elapsed
        if time.time() - self.fact_display_time > self.fact_duration:
            self.showing_fact = False
            self.phase += 1
            
            if self.phase >= len(self.phases):
                self.completed = True
        
        return frame
    
    def render_reflection(self, frame: np.ndarray) -> np.ndarray:
        """Final reflection and analysis screen"""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 10, 20), -1)
        frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)
        
        # Title
        cv2.putText(frame, 'FINANCIAL SIMULATION COMPLETE', (w//2 - 350, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 255, 180), 3)
        
        # Final stats
        self.render_stats_panel(frame, 80, 120)
        
        # Analysis panel
        analysis_x = 480
        analysis_y = 120
        analysis_width = 700
        analysis_height = 500
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (analysis_x, analysis_y),
                     (analysis_x + analysis_width, analysis_y + analysis_height),
                     (30, 40, 60), -1)
        cv2.rectangle(overlay, (analysis_x, analysis_y),
                     (analysis_x + analysis_width, analysis_y + analysis_height),
                     (100, 180, 255), 3)
        frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        
        # Analysis title
        cv2.putText(frame, 'BEHAVIOR ANALYSIS', (analysis_x + 30, analysis_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
        
        # Generate analysis
        analysis = self.generate_analysis()
        text_y = analysis_y + 90
        for line in analysis:
            cv2.putText(frame, line, (analysis_x + 30, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 240), 1)
            text_y += 30
        
        # Achievement check
        if (self.stats.stability >= 70 and 
            self.stats.liquidity >= 30 and 
            self.stats.risk < 60):
            badge_y = analysis_y + analysis_height - 100
            cv2.putText(frame, 'BALANCED FINANCIAL STRATEGY ACHIEVED',
                       (analysis_x + 100, badge_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 180), 2)
        
        # Restart button
        button_x = w // 2 - 150
        button_y = h - 100
        self.render_restart_button(frame, button_x, button_y)
        
        return frame
    
    def generate_analysis(self) -> List[str]:
        """Generate behavioral analysis based on decisions"""
        analysis = []
        
        # Risk assessment
        if self.stats.risk > 70:
            analysis.append("High Risk Profile: Your decisions favored immediate")
            analysis.append("gratification over long-term stability.")
        elif self.stats.risk < 40:
            analysis.append("Conservative Approach: You minimized risk through")
            analysis.append("careful planning and protective measures.")
        else:
            analysis.append("Balanced Risk Management: You maintained a healthy")
            analysis.append("balance between safety and opportunity.")
        
        analysis.append("")
        
        # Liquidity assessment
        if self.stats.liquidity > 60:
            analysis.append("Strong Emergency Preparedness: Your high liquidity")
            analysis.append("provided excellent crisis resilience.")
        elif self.stats.liquidity < 30:
            analysis.append("Liquidity Concerns: Low cash reserves increased")
            analysis.append("vulnerability to unexpected events.")
        
        analysis.append("")
        
        # Debt behavior
        if len(self.decision_history) > 2 and 'Credit Card EMI' in str(self.decision_history[2]):
            analysis.append("Debt Dependency: Heavy reliance on credit increased")
            analysis.append("long-term financial vulnerability.")
        
        # Insurance wisdom
        if self.insurance_level >= 1:
            analysis.append("Protective Planning: Insurance coverage demonstrated")
            analysis.append("mature risk management.")
        
        return analysis
    
    def render_restart_button(self, frame: np.ndarray, x: int, y: int):
        """Render restart button"""
        button_width = 300
        button_height = 60
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + button_width, y + button_height),
                     (50, 100, 150), -1)
        cv2.rectangle(overlay, (x, y), (x + button_width, y + button_height),
                     (100, 180, 255), 3)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        cv2.putText(frame, 'RESTART SIMULATION', (x + 40, y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 220, 255), 2)
        
        cv2.putText(frame, 'Press R to restart', (x + 70, y + button_height + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    def apply_stat_effects(self, frame: np.ndarray) -> np.ndarray:
        """Apply visual effects based on financial state"""
        h, w = frame.shape[:2]
        
        # High risk warning (red vignette)
        if self.stats.risk > 60:
            intensity = min(0.4, (self.stats.risk - 60) / 100)
            overlay = frame.copy()
            
            # Create vignette mask
            center_x, center_y = w // 2, h // 2
            for i in range(h):
                for j in range(w):
                    dist = np.sqrt((j - center_x)**2 + (i - center_y)**2)
                    max_dist = np.sqrt(center_x**2 + center_y**2)
                    vignette = (dist / max_dist) * intensity
                    overlay[i, j] = overlay[i, j] * (1 - vignette) + np.array([0, 0, 100]) * vignette
            
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Critical risk pulse
        if self.stats.risk > 75:
            pulse = 0.2 + 0.1 * np.sin(self.glow_pulse * 2)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 150), -1)
            frame = cv2.addWeighted(frame, 1 - pulse, overlay, pulse, 0)
        
        return frame
    
    def check_hover(self, hand_pos: Optional[Tuple[int, int]], 
                    x: int, y: int, w: int, h: int) -> bool:
        """Check if hand is hovering over button"""
        if hand_pos is None:
            return False
        
        hx, hy = hand_pos
        return x <= hx <= x + w and y <= hy <= y + h
    
    def update_hover(self, hand_pos: Optional[Tuple[int, int]]):
        """Update hover state and handle selection"""
        if self.phase >= len(self.phases) or self.completed or self.showing_fact:
            return
        
        # Get current decisions
        decisions = self.phases[self.phase]['decisions']
        if not decisions:
            return
        
        # Check button hovers
        h, w = 720, 1280
        button_y = h - 300
        button_width = 280
        button_spacing = 40
        total_width = len(decisions) * button_width + (len(decisions) - 1) * button_spacing
        start_x = (w - total_width) // 2
        
        current_hover = -1
        for i in range(len(decisions)):
            button_x = start_x + i * (button_width + button_spacing)
            if self.check_hover(hand_pos, button_x, button_y, button_width, 120):
                current_hover = i
                break
        
        # Update hover state
        if current_hover != self.hovered_button:
            self.hovered_button = current_hover
            self.hover_start_time = time.time()
        
        # Check for selection
        if current_hover >= 0:
            hover_time = time.time() - self.hover_start_time
            if hover_time >= self.hover_duration:
                self.apply_decision(current_hover)
                self.hovered_button = -1
    
    def apply_decision(self, decision_idx: int):
        """Apply decision and show fact"""
        if self.phase >= len(self.phases):
            return
        
        phase_data = self.phases[self.phase]
        decision = phase_data['decisions'][decision_idx]
        
        # Record decision
        self.decision_history.append(decision.title)
        
        # Track insurance level
        if self.phase == 1:
            if 'Comprehensive' in decision.title:
                self.insurance_level = 2
            elif 'Basic' in decision.title:
                self.insurance_level = 1
        
        # Special handling for investment
        if self.phase == 4:
            self.calculate_investment_outcome(decision_idx)
        
        # Apply stat changes
        for stat, change in decision.impact.items():
            current_value = getattr(self.target_stats, stat)
            setattr(self.target_stats, stat, current_value + change)
        
        # Add knowledge
        self.target_stats.knowledge += 15
        
        # Clamp stats
        self.target_stats.clamp()
        
        # Show fact
        self.current_fact = phase_data['fact']
        self.showing_fact = True
        self.fact_display_time = time.time()
    
    def alpha_blend(self, background: np.ndarray, foreground: np.ndarray, 
                    x: int, y: int) -> np.ndarray:
        """Safely blend image with alpha channel"""
        bg_h, bg_w = background.shape[:2]
        fg_h, fg_w = foreground.shape[:2]
        
        # Ensure bounds
        if x < 0 or y < 0 or x + fg_w > bg_w or y + fg_h > bg_h:
            return background
        
        # Extract alpha channel
        if foreground.shape[2] == 4:
            alpha = foreground[:, :, 3] / 255.0
            foreground_rgb = foreground[:, :, :3]
        else:
            alpha = np.ones((fg_h, fg_w))
            foreground_rgb = foreground
        
        # Blend
        roi = background[y:y+fg_h, x:x+fg_w]
        for c in range(3):
            roi[:, :, c] = (alpha * foreground_rgb[:, :, c] + 
                           (1 - alpha) * roi[:, :, c])
        
        return background
    
    def wrap_text(self, text: str, max_chars: int) -> List[str]:
        """Wrap text to multiple lines"""
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
        """Return final evaluation"""
        return {
            'stability': self.stats.stability,
            'risk': self.stats.risk,
            'liquidity': self.stats.liquidity,
            'credit_score': self.stats.credit_score,
            'knowledge': self.stats.knowledge,
            'decisions': self.decision_history,
            'balanced_strategy': (
                self.stats.stability >= 70 and 
                self.stats.liquidity >= 30 and 
                self.stats.risk < 60
            )
        }
    
    def reset(self):
        """Reset simulation to initial state"""
        self.stats = FinancialStats()
        self.target_stats = FinancialStats()
        self.phase = 0
        self.active = False
        self.completed = False
        self.hovered_button = -1
        self.hover_start_time = 0
        self.showing_fact = False
        self.decision_history = []
        self.insurance_level = 0
        self.crisis_outcome = ""
        
        # Reset crisis phase
        self.phases[3]['decisions'] = []
        self.phases[3]['context'] = None

# Global instance
_finance_sim = None

def run_finance(frame: np.ndarray, hand_pos: Optional[Tuple[int, int]]) -> np.ndarray:
    """Main entry point for finance simulation"""
    global _finance_sim
    
    if _finance_sim is None:
        _finance_sim = FinanceSimulation()
        _finance_sim.active = True
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r') and _finance_sim.completed:
        _finance_sim.reset()
        _finance_sim.active = True
    
    # Update hover state
    _finance_sim.update_hover(hand_pos)
    
    # Render
    return _finance_sim.run_finance(frame, hand_pos)

def evaluate() -> Dict:
    """Get final evaluation"""
    global _finance_sim
    if _finance_sim:
        return _finance_sim.evaluate()
    return {}

def reset():
    """Reset simulation"""
    global _finance_sim
    if _finance_sim:
        _finance_sim.reset()

# Example usage and testing
if __name__ == "__main__":
    print("Finance Simulation Module Loaded")
    print("Integration: run_finance(frame, hand_pos)")
    print("Evaluation: evaluate()")
    print("Reset: reset()")