"""
Finance Simulation - FIXED VERSION - All UI Elements Guaranteed Visible
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time
import random

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

class FinanceSimulation:
    def __init__(self):
        self.stats = FinancialStats()
        self.target_stats = FinancialStats()
        self.phase = 0
        self.active = False
        self.completed = False
        
        self.hovered_button = -1
        self.hover_start_time = 0
        self.hover_duration = 2.0
        self.showing_fact = False
        self.fact_display_time = 0
        self.fact_duration = 3.0
        self.current_fact = ""
        
        self.decision_history = []
        self.insurance_level = 0
        self.crisis_outcome = ""
        
        self.stat_lerp_speed = 0.1
        self.glow_pulse = 0.0
        
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
                    img = cv2.resize(img, (300, 300))
                    self.images[key] = img
                else:
                    self.images[key] = self.create_placeholder(key)
            except:
                self.images[key] = self.create_placeholder(key)
    
    def create_placeholder(self, label: str) -> np.ndarray:
        img = np.zeros((300, 300, 4), dtype=np.uint8)
        img[:, :, 3] = 255
        img[:, :, 0:3] = (60, 80, 100)
        cv2.putText(img, label.upper(), (20, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img
    
    def define_phases(self) -> List[Dict]:
        return [
            {
                'title': 'Housing Commitment',
                'context': 'Choose your living arrangement',
                'image': 'housing',
                'decisions': [
                    Decision('Luxury Apartment', 
                            {'stability': -15, 'risk': 15, 'liquidity': -25, 'credit_score': 10},
                            'High rent, impressive address'),
                    Decision('Shared Apartment',
                            {'stability': 10, 'risk': -5, 'liquidity': 10, 'credit_score': 5},
                            'Affordable with roommates'),
                    Decision('Stay With Parents',
                            {'stability': 15, 'risk': -10, 'liquidity': 25, 'credit_score': 0},
                            'No rent, max savings')
                ],
                'fact': 'Housing costs should stay below 30% of income'
            },
            {
                'title': 'Insurance Planning',
                'context': 'Protect against unexpected events',
                'image': 'insurance',
                'decisions': [
                    Decision('Comprehensive Insurance',
                            {'stability': 20, 'risk': -20, 'liquidity': -15, 'credit_score': 10},
                            'Full coverage protection'),
                    Decision('Basic Coverage',
                            {'stability': 10, 'risk': -5, 'liquidity': -5, 'credit_score': 5},
                            'Essential protection only'),
                    Decision('No Insurance',
                            {'stability': -10, 'risk': 25, 'liquidity': 5, 'credit_score': -5},
                            'Save money, accept risk')
                ],
                'fact': 'Medical emergencies are a leading cause of debt'
            },
            {
                'title': 'Lifestyle & Debt',
                'context': 'Your spending habits shape your future',
                'image': 'credit_card',
                'decisions': [
                    Decision('Credit Card EMI',
                            {'stability': -20, 'risk': 30, 'liquidity': 10, 'credit_score': -15},
                            'Live now, pay later'),
                    Decision('Controlled Spending',
                            {'stability': 15, 'risk': -10, 'liquidity': 5, 'credit_score': 10},
                            'Budget-conscious'),
                    Decision('Aggressive Saving',
                            {'stability': 25, 'risk': -15, 'liquidity': 20, 'credit_score': 5},
                            'Minimize expenses')
                ],
                'fact': 'High-interest debt compounds quickly if unpaid'
            },
            {
                'title': 'Crisis Event',
                'context': None,
                'image': 'crisis',
                'decisions': [],
                'fact': 'Emergency fund should cover 3-6 months expenses'
            },
            {
                'title': 'Investment Opportunity',
                'context': 'Chance to grow wealth',
                'image': 'investment',
                'decisions': [
                    Decision('Startup Investment',
                            {'stability': 0, 'risk': 25, 'liquidity': -20, 'credit_score': 0},
                            'High risk, high return'),
                    Decision('Fixed Deposit',
                            {'stability': 15, 'risk': -10, 'liquidity': -10, 'credit_score': 10},
                            'Safe, steady returns'),
                    Decision('No Investment',
                            {'stability': -5, 'risk': 0, 'liquidity': 5, 'credit_score': 0},
                            'Keep cash liquid')
                ],
                'fact': 'Diversification reduces investment risk'
            }
        ]
    
    def generate_crisis(self) -> Dict:
        crises = [
            {'type': 'Medical Emergency', 'desc': 'Unexpected hospitalization'},
            {'type': 'Job Loss', 'desc': 'Company downsizing'},
            {'type': 'Vehicle Breakdown', 'desc': 'Critical repairs needed'}
        ]
        
        crisis = random.choice(crises)
        
        base_stability_loss = 25
        base_liquidity_loss = 30
        base_risk_increase = 20
        
        if self.insurance_level == 2:
            base_stability_loss *= 0.3
            base_liquidity_loss *= 0.4
            base_risk_increase *= 0.2
            crisis['outcome'] = 'Insurance covered costs'
        elif self.insurance_level == 1:
            base_stability_loss *= 0.6
            base_liquidity_loss *= 0.7
            base_risk_increase *= 0.5
            crisis['outcome'] = 'Partial insurance help'
        else:
            crisis['outcome'] = 'Full impact absorbed'
        
        if self.stats.liquidity > 60:
            base_stability_loss *= 0.7
        elif self.stats.liquidity < 30:
            base_stability_loss *= 1.3
            base_risk_increase *= 1.2
        
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
                self.current_fact = 'Investment succeeded!'
            else:
                decision.impact = {'stability': -25, 'risk': 15, 'liquidity': -20, 'credit_score': -10}
                self.current_fact = 'Investment failed'
    
    def run_finance(self, frame: np.ndarray, hand_pos: Optional[Tuple[int, int]]) -> np.ndarray:
        if not self.active:
            return frame
        
        self.update_animations()
        self.lerp_stats()
        
        if self.showing_fact:
            frame = self.render_fact_screen(frame)
        elif self.completed:
            frame = self.render_reflection(frame)
        else:
            frame = self.render_phase(frame, hand_pos)
        
        return frame
    
    def update_animations(self):
        self.glow_pulse = (self.glow_pulse + 0.08) % (2 * np.pi)
    
    def lerp_stats(self):
        self.stats.stability += (self.target_stats.stability - self.stats.stability) * self.stat_lerp_speed
        self.stats.risk += (self.target_stats.risk - self.stats.risk) * self.stat_lerp_speed
        self.stats.liquidity += (self.target_stats.liquidity - self.stats.liquidity) * self.stat_lerp_speed
        self.stats.credit_score += (self.target_stats.credit_score - self.stats.credit_score) * self.stat_lerp_speed
        self.stats.knowledge += (self.target_stats.knowledge - self.stats.knowledge) * self.stat_lerp_speed
    
    def render_phase(self, frame: np.ndarray, hand_pos: Optional[Tuple[int, int]]) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Light overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Generate crisis
        if self.phase == 3 and not self.phases[3]['decisions']:
            crisis = self.generate_crisis()
            self.phases[3]['context'] = crisis['desc']
            self.phases[3]['decisions'] = [
                Decision('Accept Impact', crisis['impact'], crisis['outcome'])
            ]
        
        phase_data = self.phases[self.phase]
        
        # Title
        cv2.putText(frame, phase_data['title'], (w//2 - 250, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 220, 150), 3)
        
        # Stats panel LEFT
        self.render_stats_panel(frame, 40, 120)
        
        # Context CENTER
        if phase_data['context']:
            ctx_x = w//2 - 180
            ctx_y = 120
            # Solid background
            cv2.rectangle(frame, (ctx_x, ctx_y), (ctx_x + 360, ctx_y + 70), (15, 20, 30), -1)
            cv2.rectangle(frame, (ctx_x, ctx_y), (ctx_x + 360, ctx_y + 70), (100, 200, 255), 3)
            cv2.putText(frame, phase_data['context'], (ctx_x + 15, ctx_y + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        # Image RIGHT
        img_x = w - 350
        img_y = 120
        if phase_data['image'] in self.images:
            img = self.images[phase_data['image']]
            # Border
            cv2.rectangle(frame, (img_x - 3, img_y - 3), (img_x + 303, img_y + 303), (100, 200, 255), 4)
            # Draw image
            frame = self.alpha_blend(frame, img, img_x, img_y)
        
        # BUTTONS AT BOTTOM
        button_y = h - 150
        decisions = phase_data['decisions']
        
        if len(decisions) > 0:
            button_width = 260
            button_spacing = 60
            total_width = len(decisions) * button_width + (len(decisions) - 1) * button_spacing
            start_x = (w - total_width) // 2
            
            for i, decision in enumerate(decisions):
                button_x = start_x + i * (button_width + button_spacing)
                is_hovered = self.check_hover(hand_pos, button_x, button_y, button_width, 120)
                
                # SOLID button background
                bg_color = (40, 60, 80) if is_hovered else (25, 35, 50)
                cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + 120), bg_color, -1)
                
                # BRIGHT border
                border_color = (150, 255, 255) if is_hovered else (100, 200, 255)
                thickness = 5 if is_hovered else 3
                cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + 120), border_color, thickness)
                
                # Title
                cv2.putText(frame, decision.title, (button_x + 15, button_y + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Description
                cv2.putText(frame, decision.description, (button_x + 15, button_y + 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1)
                
                # Progress bar
                if is_hovered and hand_pos:
                    progress = min(1.0, (time.time() - self.hover_start_time) / self.hover_duration)
                    bar_width = int(progress * (button_width - 20))
                    cv2.rectangle(frame, (button_x + 10, button_y + 100),
                                 (button_x + 10 + bar_width, button_y + 110),
                                 (150, 255, 200), -1)
                    cv2.putText(frame, f'{int(progress * 100)}%', (button_x + button_width//2 - 25, button_y + 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Preview panel
        if self.hovered_button >= 0 and self.hovered_button < len(decisions):
            self.render_preview_panel(frame, w//2 - 200, button_y - 160, decisions[self.hovered_button])
        
        return frame
    
    def render_stats_panel(self, frame: np.ndarray, x: int, y: int):
        panel_width = 280
        panel_height = 340
        
        # SOLID background
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (15, 20, 30), -1)
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (100, 200, 255), 4)
        
        # Title
        cv2.putText(frame, 'FINANCIAL STATUS', (x + 15, y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Stats
        stats_list = [
            ('Stability', self.stats.stability, (120, 255, 120)),
            ('Risk', self.stats.risk, (120, 150, 255)),
            ('Liquidity', self.stats.liquidity, (255, 200, 120)),
            ('Credit', self.stats.credit_score, (255, 150, 200))
        ]
        
        bar_y = y + 70
        for label, value, color in stats_list:
            # Label
            cv2.putText(frame, label, (x + 15, bar_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Value
            cv2.putText(frame, f'{int(value)}', (x + panel_width - 50, bar_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Bar
            bar_start_y = bar_y + 10
            cv2.rectangle(frame, (x + 15, bar_start_y), (x + panel_width - 15, bar_start_y + 20), (30, 35, 40), -1)
            cv2.rectangle(frame, (x + 15, bar_start_y), (x + panel_width - 15, bar_start_y + 20), (80, 80, 80), 2)
            
            fill_width = int((value / 100.0) * (panel_width - 34))
            if fill_width > 0:
                cv2.rectangle(frame, (x + 17, bar_start_y + 2), (x + 17 + fill_width, bar_start_y + 18), color, -1)
            
            bar_y += 70
    
    def render_preview_panel(self, frame: np.ndarray, x: int, y: int, decision: Decision):
        panel_width = 400
        panel_height = 150
        
        # SOLID background
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (20, 30, 40), -1)
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (255, 200, 100), 4)
        
        # Title
        cv2.putText(frame, 'PROJECTED IMPACT', (x + 20, y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 150), 2)
        
        # Impacts
        text_y = y + 65
        for stat, change in decision.impact.items():
            if change == 0:
                continue
            
            label = stat.replace('_', ' ').title()
            sign = '+' if change > 0 else ''
            color = (150, 255, 150) if change > 0 else (150, 150, 255)
            arrow = '↑' if change > 0 else '↓'
            
            text = f"{arrow} {label}: {sign}{int(change)}"
            cv2.putText(frame, text, (x + 30, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            text_y += 28
    
    def render_fact_screen(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Fact panel
        panel_width = 800
        panel_height = 220
        panel_x = (w - panel_width) // 2
        panel_y = (h - panel_height) // 2
        
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (20, 30, 40), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (150, 255, 200), 5)
        
        cv2.putText(frame, 'FINANCIAL INSIGHT', (panel_x + 40, panel_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (150, 255, 200), 3)
        
        cv2.putText(frame, self.current_fact, (panel_x + 40, panel_y + 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if time.time() - self.fact_display_time > self.fact_duration:
            self.showing_fact = False
            self.phase += 1
            if self.phase >= len(self.phases):
                self.completed = True
        
        return frame
    
    def render_reflection(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 5, 10), -1)
        frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)
        
        cv2.putText(frame, 'SIMULATION COMPLETE', (w//2 - 300, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.4, (150, 255, 200), 3)
        
        self.render_stats_panel(frame, 60, 120)
        
        # Analysis
        analysis_x = 400
        analysis_y = 120
        analysis_width = 800
        analysis_height = 400
        
        cv2.rectangle(frame, (analysis_x, analysis_y), (analysis_x + analysis_width, analysis_y + analysis_height), (15, 20, 30), -1)
        cv2.rectangle(frame, (analysis_x, analysis_y), (analysis_x + analysis_width, analysis_y + analysis_height), (100, 200, 255), 4)
        
        cv2.putText(frame, 'BEHAVIOR ANALYSIS', (analysis_x + 30, analysis_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        analysis = self.generate_analysis()
        text_y = analysis_y + 90
        for line in analysis:
            cv2.putText(frame, line, (analysis_x + 30, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 230, 255), 1)
            text_y += 32
        
        # Achievement
        if (self.stats.stability >= 70 and self.stats.liquidity >= 30 and self.stats.risk < 60):
            cv2.putText(frame, 'BALANCED STRATEGY ACHIEVED', (analysis_x + 150, analysis_y + analysis_height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.85, (150, 255, 150), 2)
        
        # Restart
        button_x = w // 2 - 150
        button_y = h - 100
        cv2.rectangle(frame, (button_x, button_y), (button_x + 300, button_y + 60), (30, 50, 70), -1)
        cv2.rectangle(frame, (button_x, button_y), (button_x + 300, button_y + 60), (100, 200, 255), 4)
        cv2.putText(frame, 'Press R to Restart', (button_x + 40, button_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def generate_analysis(self) -> List[str]:
        analysis = []
        
        if self.stats.risk > 70:
            analysis.append("High Risk: Favored immediate gratification")
        elif self.stats.risk < 40:
            analysis.append("Conservative: Minimized risk carefully")
        else:
            analysis.append("Balanced: Healthy risk-safety balance")
        
        analysis.append("")
        
        if self.stats.liquidity > 60:
            analysis.append("Strong Liquidity: Excellent preparedness")
        elif self.stats.liquidity < 30:
            analysis.append("Low Liquidity: Vulnerable to crises")
        
        analysis.append("")
        
        if self.insurance_level >= 1:
            analysis.append("Protected: Insurance showed maturity")
        
        return analysis
    
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
        button_y = h - 150
        button_width = 260
        button_spacing = 60
        total_width = len(decisions) * button_width + (len(decisions) - 1) * button_spacing
        start_x = (w - total_width) // 2
        
        current_hover = -1
        for i in range(len(decisions)):
            button_x = start_x + i * (button_width + button_spacing)
            if self.check_hover(hand_pos, button_x, button_y, button_width, 120):
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
            elif 'Basic' in decision.title:
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
        self.phase = 0
        self.active = False
        self.completed = False
        self.hovered_button = -1
        self.hover_start_time = 0
        self.showing_fact = False
        self.decision_history = []
        self.insurance_level = 0
        self.crisis_outcome = ""
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
    print("Finance Module - FIXED - All UI Elements Visible")