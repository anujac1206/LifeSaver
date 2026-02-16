"""
Finance Simulation - MAXIMUM VISIBILITY EDITION
Ultra-bright UI optimized for webcam overlay with 3-second selection time
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
        self.hover_duration = 3.0  # 3 SECONDS
        self.showing_fact = False
        self.fact_display_time = 0
        self.fact_duration = 3.0
        self.current_fact = ""
        
        self.decision_history = []
        self.insurance_level = 0
        self.crisis_outcome = ""
        
        self.stat_lerp_speed = 0.08
        
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
                    img = cv2.resize(img, (280, 280))
                    self.images[key] = img
                else:
                    self.images[key] = self.create_placeholder(key)
            except:
                self.images[key] = self.create_placeholder(key)
    
    def create_placeholder(self, label: str) -> np.ndarray:
        img = np.zeros((280, 280, 4), dtype=np.uint8)
        img[:, :, 3] = 255
        img[:, :, 0:3] = (80, 100, 120)
        cv2.putText(img, label.upper(), (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img
    
    def define_phases(self) -> List[Dict]:
        return [
            {
                'title': 'Housing Commitment',
                'context': 'Choose your living situation',
                'image': 'housing',
                'decisions': [
                    Decision('Luxury Apartment', 
                            {'stability': -15, 'risk': 15, 'liquidity': -25, 'credit_score': 10},
                            'High rent, premium location'),
                    Decision('Shared Living',
                            {'stability': 10, 'risk': -5, 'liquidity': 10, 'credit_score': 5},
                            'Affordable roommate setup'),
                    Decision('Family Support',
                            {'stability': 15, 'risk': -10, 'liquidity': 25, 'credit_score': 0},
                            'No rent, max savings')
                ],
                'fact': 'Housing costs should stay below 30% of income'
            },
            {
                'title': 'Insurance Strategy',
                'context': 'Protect against uncertainty',
                'image': 'insurance',
                'decisions': [
                    Decision('Comprehensive Coverage',
                            {'stability': 20, 'risk': -20, 'liquidity': -15, 'credit_score': 10},
                            'Full protection package'),
                    Decision('Essential Coverage',
                            {'stability': 10, 'risk': -5, 'liquidity': -5, 'credit_score': 5},
                            'Core protection only'),
                    Decision('Self-Insurance',
                            {'stability': -10, 'risk': 25, 'liquidity': 5, 'credit_score': -5},
                            'Accept all risks')
                ],
                'fact': 'Medical emergencies cause 60% of bankruptcies'
            },
            {
                'title': 'Lifestyle Choice',
                'context': 'Balance spending and saving',
                'image': 'credit_card',
                'decisions': [
                    Decision('Credit Card Lifestyle',
                            {'stability': -20, 'risk': 30, 'liquidity': 10, 'credit_score': -15},
                            'Finance present consumption'),
                    Decision('Mindful Spending',
                            {'stability': 15, 'risk': -10, 'liquidity': 5, 'credit_score': 10},
                            'Strategic allocation'),
                    Decision('Extreme Frugality',
                            {'stability': 25, 'risk': -15, 'liquidity': 20, 'credit_score': 5},
                            'Minimize all expenses')
                ],
                'fact': 'High-interest debt compounds 5-10x faster than investments'
            },
            {
                'title': 'Crisis Event',
                'context': None,
                'image': 'crisis',
                'decisions': [],
                'fact': '6 months savings = 3x better recovery from shocks'
            },
            {
                'title': 'Investment Decision',
                'context': 'Growth vs security tradeoff',
                'image': 'investment',
                'decisions': [
                    Decision('Venture Capital',
                            {'stability': 0, 'risk': 25, 'liquidity': -20, 'credit_score': 0},
                            'High risk, high reward'),
                    Decision('Fixed Income',
                            {'stability': 15, 'risk': -10, 'liquidity': -10, 'credit_score': 10},
                            'Guaranteed safe returns'),
                    Decision('Cash Position',
                            {'stability': -5, 'risk': 0, 'liquidity': 5, 'credit_score': 0},
                            'Maximum flexibility')
                ],
                'fact': 'Diversified portfolios reduce volatility by 40%'
            }
        ]
    
    def generate_crisis(self) -> Dict:
        crises = [
            {'type': 'Medical Emergency', 'desc': 'Unexpected hospitalization'},
            {'type': 'Job Loss', 'desc': 'Income stopped for 3 months'},
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
            crisis['outcome'] = 'Insurance covered most costs'
        elif self.insurance_level == 1:
            base_stability_loss *= 0.6
            base_liquidity_loss *= 0.7
            base_risk_increase *= 0.5
            crisis['outcome'] = 'Partial insurance coverage'
        else:
            crisis['outcome'] = 'Full financial impact'
        
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
                self.current_fact = 'Investment succeeded! Strong fundamentals paid off.'
            else:
                decision.impact = {'stability': -25, 'risk': 15, 'liquidity': -20, 'credit_score': -10}
                self.current_fact = 'Investment failed. Only invest what you can lose.'
    
    def run_finance(self, frame: np.ndarray, hand_pos: Optional[Tuple[int, int]]) -> np.ndarray:
        if not self.active:
            return frame
        
        self.lerp_stats()
        
        if self.showing_fact:
            frame = self.render_fact_screen(frame)
        elif self.completed:
            frame = self.render_reflection(frame)
        else:
            frame = self.render_phase(frame, hand_pos)
        
        return frame
    
    def lerp_stats(self):
        self.stats.stability += (self.target_stats.stability - self.stats.stability) * self.stat_lerp_speed
        self.stats.risk += (self.target_stats.risk - self.stats.risk) * self.stat_lerp_speed
        self.stats.liquidity += (self.target_stats.liquidity - self.stats.liquidity) * self.stat_lerp_speed
        self.stats.credit_score += (self.target_stats.credit_score - self.stats.credit_score) * self.stat_lerp_speed
        self.stats.knowledge += (self.target_stats.knowledge - self.stats.knowledge) * self.stat_lerp_speed
    
    def render_phase(self, frame: np.ndarray, hand_pos: Optional[Tuple[int, int]]) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # MINIMAL overlay - keep webcam visible
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.80, overlay, 0.20, 0)  # Only 20% dark
        
        # Generate crisis
        if self.phase == 3 and not self.phases[3]['decisions']:
            crisis = self.generate_crisis()
            self.phases[3]['context'] = crisis['desc']
            self.phases[3]['decisions'] = [
                Decision('Accept Impact', crisis['impact'], crisis['outcome'])
            ]
        
        phase_data = self.phases[self.phase]
        
        # ===== GIANT BRIGHT HEADING AT TOP =====
        title = phase_data['title']
        title_scale = 2.0
        title_thickness = 4
        
        # Get text size for centering
        (title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thickness)
        title_x = (w - title_w) // 2
        title_y = 80
        
        # Black background for title
        padding = 25
        cv2.rectangle(frame, (title_x - padding, title_y - title_h - padding),
                     (title_x + title_w + padding, title_y + padding),
                     (0, 0, 0), -1)
        
        # BRIGHT YELLOW title
        cv2.putText(frame, title, (title_x, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 255, 255), title_thickness)
        
        # ===== THREE COLUMN LAYOUT =====
        col_width = w // 3
        content_y = 140
        
        # LEFT: Stats
        self.render_ultra_bright_stats(frame, 30, content_y, col_width - 60)
        
        # CENTER: Context + Image
        center_x = col_width + 30
        if phase_data['context']:
            # Context text
            ctx_lines = self.wrap_text(phase_data['context'], 30)
            ctx_y = content_y + 20
            for line in ctx_lines:
                # Black background
                (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (center_x - 5, ctx_y - text_h - 5),
                             (center_x + text_w + 5, ctx_y + 5), (0, 0, 0), -1)
                # White text
                cv2.putText(frame, line, (center_x, ctx_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ctx_y += 35
        
        # Image
        if phase_data['image'] in self.images:
            img_x = center_x + 20
            img_y = content_y + 100
            img = self.images[phase_data['image']]
            
            # Bright border
            border_size = 5
            cv2.rectangle(frame, (img_x - border_size, img_y - border_size),
                         (img_x + 280 + border_size, img_y + 280 + border_size),
                         (0, 255, 255), border_size)
            
            frame = self.alpha_blend(frame, img, img_x, img_y)
        
        # RIGHT: Preview or guidance
        right_x = 2 * col_width + 30
        decisions = phase_data['decisions']
        
        if self.hovered_button >= 0 and self.hovered_button < len(decisions):
            self.render_ultra_bright_preview(frame, right_x, content_y, col_width - 60, 
                                            decisions[self.hovered_button])
        else:
            # Guidance text
            guide_y = content_y + 30
            guide_lines = ['Hover over', 'choices below', '', 'Hold 3 seconds', 'to confirm']
            for line in guide_lines:
                (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (right_x - 5, guide_y - text_h - 5),
                             (right_x + text_w + 5, guide_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, line, (right_x, guide_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                guide_y += 40
        
        # ===== DECISION BUTTONS AT BOTTOM =====
        button_y = h - 160
        
        if len(decisions) > 0:
            button_width = 340
            button_spacing = 50
            total_width = len(decisions) * button_width + (len(decisions) - 1) * button_spacing
            start_x = (w - total_width) // 2
            
            for i, decision in enumerate(decisions):
                button_x = start_x + i * (button_width + button_spacing)
                is_hovered = self.check_hover(hand_pos, button_x, button_y, button_width, 130)
                
                self.render_ultra_bright_button(frame, button_x, button_y, button_width, 130,
                                               decision, i, is_hovered, hand_pos)
        
        return frame
    
    def render_ultra_bright_stats(self, frame: np.ndarray, x: int, y: int, width: int):
        """ULTRA BRIGHT stats panel"""
        panel_height = 360
        
        # BLACK background
        cv2.rectangle(frame, (x, y), (x + width, y + panel_height), (0, 0, 0), -1)
        
        # BRIGHT CYAN border
        cv2.rectangle(frame, (x, y), (x + width, y + panel_height), (0, 255, 255), 4)
        
        # Title
        cv2.putText(frame, 'FINANCIAL STATUS', (x + 15, y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        stats_list = [
            ('Stability', self.stats.stability, (0, 255, 0)),      # GREEN
            ('Risk', self.stats.risk, (255, 100, 255)),            # MAGENTA
            ('Liquidity', self.stats.liquidity, (0, 255, 255)),    # CYAN
            ('Credit', self.stats.credit_score, (255, 255, 0))     # YELLOW
        ]
        
        stat_y = y + 70
        for label, value, color in stats_list:
            # Label
            cv2.putText(frame, label, (x + 15, stat_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            
            # Value - HUGE
            value_text = f'{int(value)}'
            cv2.putText(frame, value_text, (x + width - 60, stat_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
            
            # Bar
            bar_y = stat_y + 10
            bar_width = width - 30
            
            # Background
            cv2.rectangle(frame, (x + 15, bar_y), (x + 15 + bar_width, bar_y + 18), (40, 40, 40), -1)
            
            # Fill
            fill = int((value / 100.0) * bar_width)
            if fill > 0:
                cv2.rectangle(frame, (x + 15, bar_y), (x + 15 + fill, bar_y + 18), color, -1)
            
            stat_y += 80
    
    def render_ultra_bright_preview(self, frame: np.ndarray, x: int, y: int, width: int, decision: Decision):
        """ULTRA BRIGHT preview panel"""
        height = 280
        
        # BLACK background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), -1)
        
        # BRIGHT YELLOW border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 5)
        
        # Title
        cv2.putText(frame, 'IMPACT:', (x + 20, y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Impacts
        impact_y = y + 80
        for stat, change in decision.impact.items():
            if change == 0:
                continue
            
            label = stat.replace('_', ' ').title()
            sign = '+' if change > 0 else ''
            color = (0, 255, 0) if change > 0 else (0, 100, 255)  # GREEN or RED
            arrow = '↑' if change > 0 else '↓'
            
            text = f"{arrow} {label}"
            cv2.putText(frame, text, (x + 25, impact_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            
            change_text = f'{sign}{int(change)}'
            cv2.putText(frame, change_text, (x + width - 70, impact_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
            
            impact_y += 45
    
    def render_ultra_bright_button(self, frame: np.ndarray, x: int, y: int, w: int, h: int,
                                   decision: Decision, idx: int, is_hovered: bool, 
                                   hand_pos: Optional[Tuple[int, int]]):
        """ULTRA BRIGHT decision buttons"""
        # BLACK background
        bg_color = (20, 20, 20) if is_hovered else (10, 10, 10)
        cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)
        
        # BRIGHT CYAN/YELLOW border
        border_color = (0, 255, 255) if is_hovered else (100, 255, 255)
        border_thickness = 6 if is_hovered else 4
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, border_thickness)
        
        # Title - WHITE
        title_lines = self.wrap_text(decision.title, 18)
        title_y = y + 35
        for line in title_lines:
            cv2.putText(frame, line, (x + 20, title_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            title_y += 30
        
        # Description - CYAN
        desc_lines = self.wrap_text(decision.description, 24)
        desc_y = title_y + 10
        for line in desc_lines:
            cv2.putText(frame, line, (x + 20, desc_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
            desc_y += 24
        
        # Progress bar - 3 SECONDS
        if is_hovered and hand_pos:
            progress = min(1.0, (time.time() - self.hover_start_time) / self.hover_duration)
            
            bar_y = y + h - 22
            bar_width = w - 40
            
            # Background
            cv2.rectangle(frame, (x + 20, bar_y), (x + 20 + bar_width, bar_y + 12), (40, 40, 40), -1)
            
            # Fill - BRIGHT GREEN
            fill = int(progress * bar_width)
            if fill > 0:
                cv2.rectangle(frame, (x + 20, bar_y), (x + 20 + fill, bar_y + 12), (0, 255, 0), -1)
            
            # Percentage - HUGE
            percent_text = f'{int(progress * 100)}%'
            cv2.putText(frame, percent_text, (x + w//2 - 30, bar_y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def render_fact_screen(self, frame: np.ndarray) -> np.ndarray:
        """ULTRA BRIGHT fact screen"""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)
        
        # Panel
        panel_width = 900
        panel_height = 280
        panel_x = (w - panel_width) // 2
        panel_y = (h - panel_height) // 2
        
        # BLACK background
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        
        # BRIGHT YELLOW border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 255, 255), 6)
        
        # Title
        cv2.putText(frame, 'KEY INSIGHT', (panel_x + panel_width//2 - 130, panel_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        
        # Fact
        fact_lines = self.wrap_text(self.current_fact, 60)
        fact_y = panel_y + 130
        for line in fact_lines:
            (text_w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(frame, line, (panel_x + (panel_width - text_w)//2, fact_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            fact_y += 40
        
        if time.time() - self.fact_display_time > self.fact_duration:
            self.showing_fact = False
            self.phase += 1
            if self.phase >= len(self.phases):
                self.completed = True
        
        return frame
    
    def render_reflection(self, frame: np.ndarray) -> np.ndarray:
        """ULTRA BRIGHT completion screen"""
        h, w = frame.shape[:2]
        
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.15, overlay, 0.85, 0)
        
        # Title
        title = 'SIMULATION COMPLETE'
        title_scale = 1.8
        (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 3)
        cv2.rectangle(frame, (w//2 - title_w//2 - 20, 30), (w//2 + title_w//2 + 20, 90), (0, 0, 0), -1)
        cv2.putText(frame, title, (w//2 - title_w//2, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 255, 255), 3)
        
        # Stats
        self.render_ultra_bright_stats(frame, 50, 130, 340)
        
        # Analysis
        analysis_x = 450
        analysis_y = 130
        analysis_width = 780
        analysis_height = 380
        
        cv2.rectangle(frame, (analysis_x, analysis_y), (analysis_x + analysis_width, analysis_y + analysis_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (analysis_x, analysis_y), (analysis_x + analysis_width, analysis_y + analysis_height), (0, 255, 255), 4)
        
        cv2.putText(frame, 'ANALYSIS', (analysis_x + 30, analysis_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        
        analysis = self.generate_analysis()
        text_y = analysis_y + 90
        for line in analysis:
            cv2.putText(frame, line, (analysis_x + 30, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            text_y += 32
        
        # Achievement
        if (self.stats.stability >= 70 and self.stats.liquidity >= 30 and self.stats.risk < 60):
            cv2.putText(frame, 'BALANCED STRATEGY!', (analysis_x + 220, analysis_y + analysis_height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Restart
        button_x = w // 2 - 180
        button_y = h - 110
        cv2.rectangle(frame, (button_x, button_y), (button_x + 360, button_y + 70), (0, 0, 0), -1)
        cv2.rectangle(frame, (button_x, button_y), (button_x + 360, button_y + 70), (0, 255, 255), 4)
        cv2.putText(frame, 'Press R to Restart', (button_x + 50, button_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        return frame
    
    def generate_analysis(self) -> List[str]:
        analysis = []
        
        if self.stats.risk > 70:
            analysis.append("High Risk: Prioritized short-term gains")
        elif self.stats.risk < 40:
            analysis.append("Conservative: Minimized risk exposure")
        else:
            analysis.append("Balanced: Good risk-safety equilibrium")
        
        analysis.append("")
        
        if self.stats.liquidity > 60:
            analysis.append("Strong Liquidity: Excellent reserves")
        elif self.stats.liquidity < 30:
            analysis.append("Low Liquidity: Vulnerable to shocks")
        
        analysis.append("")
        
        if self.insurance_level >= 1:
            analysis.append("Protected: Insurance coverage active")
        
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
        button_y = h - 160
        button_width = 340
        button_spacing = 50
        total_width = len(decisions) * button_width + (len(decisions) - 1) * button_spacing
        start_x = (w - total_width) // 2
        
        current_hover = -1
        for i in range(len(decisions)):
            button_x = start_x + i * (button_width + button_spacing)
            if self.check_hover(hand_pos, button_x, button_y, button_width, 130):
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
    print("Finance Module - MAXIMUM VISIBILITY")
    print("• 3 second selection time")
    print("• Ultra-bright colors")
    print("• Giant heading")
    print("• Black backgrounds with bright borders")