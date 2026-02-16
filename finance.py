"""
Finance Simulation - PROFESSIONAL HIGH VISIBILITY EDITION
Sophisticated design with maximum readability over webcam
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
        
        # PROFESSIONAL COLOR PALETTE
        self.colors = {
            'slate_900': (15, 23, 42),           # Deep professional background
            'slate_800': (30, 41, 59),           # Card backgrounds
            'slate_700': (51, 65, 85),           # Borders
            'blue_400': (251, 191, 96),          # Elegant gold accent
            'blue_500': (255, 171, 59),          # Primary blue - bright but professional
            'blue_600': (231, 146, 37),          # Deeper blue
            'emerald_400': (178, 245, 129),      # Success green
            'rose_400': (112, 118, 251),         # Professional red
            'white': (255, 255, 255),            # Pure white
            'gray_100': (243, 244, 246),         # Light gray
            'gray_300': (209, 213, 219),         # Medium gray
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
                    img = cv2.resize(img, (280, 280))
                    self.images[key] = img
                else:
                    self.images[key] = self.create_placeholder(key)
            except:
                self.images[key] = self.create_placeholder(key)
    
    def create_placeholder(self, label: str) -> np.ndarray:
        img = np.zeros((280, 280, 4), dtype=np.uint8)
        img[:, :, 3] = 255
        
        # Professional gradient
        for i in range(280):
            intensity = int(50 + (i / 280) * 30)
            img[i, :, 0] = intensity
            img[i, :, 1] = intensity + 10
            img[i, :, 2] = intensity + 15
        
        cv2.putText(img, label.upper(), (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['white'], 2)
        return img
    
    def define_phases(self) -> List[Dict]:
        return [
            {
                'title': 'Housing Commitment',
                'context': 'Your living situation significantly impacts monthly cash flow',
                'image': 'housing',
                'decisions': [
                    Decision('Luxury Apartment', 
                            {'stability': -15, 'risk': 15, 'liquidity': -25, 'credit_score': 10},
                            'Premium location, high monthly cost'),
                    Decision('Shared Living',
                            {'stability': 10, 'risk': -5, 'liquidity': 10, 'credit_score': 5},
                            'Affordable with roommates'),
                    Decision('Family Support',
                            {'stability': 15, 'risk': -10, 'liquidity': 25, 'credit_score': 0},
                            'Zero rent, maximum savings')
                ],
                'fact': 'Financial advisors recommend keeping housing costs below 30% of gross income for optimal wealth building.'
            },
            {
                'title': 'Insurance Strategy',
                'context': 'Insurance trades predictable costs for protection against catastrophic losses',
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
                'fact': 'Medical emergencies account for over 60% of personal bankruptcies in developed nations.'
            },
            {
                'title': 'Lifestyle Calibration',
                'context': 'Every dollar spent today cannot compound tomorrow',
                'image': 'credit_card',
                'decisions': [
                    Decision('Credit Card Lifestyle',
                            {'stability': -20, 'risk': 30, 'liquidity': 10, 'credit_score': -15},
                            'Finance present consumption'),
                    Decision('Mindful Spending',
                            {'stability': 15, 'risk': -10, 'liquidity': 5, 'credit_score': 10},
                            'Strategic allocation with discipline'),
                    Decision('Extreme Frugality',
                            {'stability': 25, 'risk': -15, 'liquidity': 20, 'credit_score': 5},
                            'Minimize all non-essential expenses')
                ],
                'fact': 'High-interest consumer debt compounds at rates 5-10x higher than typical investment returns.'
            },
            {
                'title': 'Crisis Response',
                'context': None,
                'image': 'crisis',
                'decisions': [],
                'fact': 'Individuals with 6 months of expenses saved are 3x more likely to recover from financial shocks without lasting damage.'
            },
            {
                'title': 'Capital Deployment',
                'context': 'Investment decisions should align with risk capacity and time horizon',
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
                            'Maintain maximum flexibility')
                ],
                'fact': 'Diversified portfolios historically reduce volatility by 40% while maintaining 90% of concentrated portfolio returns.'
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
            crisis['outcome'] = 'Insurance coverage absorbed majority of impact'
        elif self.insurance_level == 1:
            base_stability_loss *= 0.6
            base_liquidity_loss *= 0.7
            base_risk_increase *= 0.5
            crisis['outcome'] = 'Partial coverage provided, significant out-of-pocket required'
        else:
            crisis['outcome'] = 'Full financial impact absorbed directly'
        
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
                self.current_fact = 'Investment succeeded. Strong financial fundamentals enabled capital growth opportunity.'
            else:
                decision.impact = {'stability': -25, 'risk': 15, 'liquidity': -20, 'credit_score': -10}
                self.current_fact = 'Investment underperformed. Weak financial foundation amplified losses. Only invest what you can afford to lose entirely.'
    
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
        
        # Professional subtle overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), self.colors['slate_900'], -1)
        frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)
        
        # Generate crisis
        if self.phase == 3 and not self.phases[3]['decisions']:
            crisis = self.generate_crisis()
            self.phases[3]['context'] = crisis['desc']
            self.phases[3]['decisions'] = [
                Decision('Accept Impact', crisis['impact'], crisis['outcome'])
            ]
        
        phase_data = self.phases[self.phase]
        
        # ===== ELEGANT HEADER =====
        title = phase_data['title']
        title_scale = 1.8
        title_thickness = 3
        
        (title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thickness)
        title_x = (w - title_w) // 2
        title_y = 75
        
        # Elegant background panel
        panel_padding = 30
        cv2.rectangle(frame, (title_x - panel_padding, title_y - title_h - 15),
                     (title_x + title_w + panel_padding, title_y + 15),
                     self.colors['slate_800'], -1)
        cv2.rectangle(frame, (title_x - panel_padding, title_y - title_h - 15),
                     (title_x + title_w + panel_padding, title_y + 15),
                     self.colors['blue_500'], 2)
        
        # Professional white title
        cv2.putText(frame, title, (title_x, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, title_scale, self.colors['white'], title_thickness, cv2.LINE_AA)
        
        # Subtle accent line
        line_y = title_y + 25
        cv2.line(frame, (title_x, line_y), (title_x + title_w, line_y), self.colors['blue_500'], 2)
        
        # ===== THREE COLUMN LAYOUT =====
        col_width = w // 3
        content_y = 140
        
        # LEFT: Stats
        self.render_professional_stats(frame, 30, content_y, col_width - 60)
        
        # CENTER: Context + Image
        center_x = col_width + 30
        if phase_data['context']:
            self.render_professional_context(frame, center_x, content_y, col_width - 60, phase_data['context'])
        
        # Image
        if phase_data['image'] in self.images:
            img_x = center_x + 20
            img_y = content_y + 110
            img = self.images[phase_data['image']]
            
            # Professional frame
            # cv2.rectangle(frame, (img_x - 4, img_y - 4),
            #              (img_x + 284, img_y + 284),
            #              self.colors['blue_500'], 3, cv2.LINE_AA)
            
            frame = self.alpha_blend(frame, img, img_x, img_y)
        
        # RIGHT: Preview or guidance
        right_x = 2 * col_width + 30
        decisions = phase_data['decisions']
        
        if self.hovered_button >= 0 and self.hovered_button < len(decisions):
            self.render_professional_preview(frame, right_x, content_y, col_width - 60, 
                                            decisions[self.hovered_button])
        else:
            self.render_professional_guidance(frame, right_x, content_y, col_width - 60)
        
        # ===== DECISION CARDS AT BOTTOM =====
        button_y = h - 165
        
        if len(decisions) > 0:
            card_width = 340
            card_spacing = 50
            total_width = len(decisions) * card_width + (len(decisions) - 1) * card_spacing
            start_x = (w - total_width) // 2
            
            for i, decision in enumerate(decisions):
                card_x = start_x + i * (card_width + card_spacing)
                is_hovered = self.check_hover(hand_pos, card_x, button_y, card_width, 140)
                
                self.render_professional_card(frame, card_x, button_y, card_width, 140,
                                             decision, i, is_hovered, hand_pos)
        
        return frame
    
    def render_professional_stats(self, frame: np.ndarray, x: int, y: int, width: int):
        """Professional stats panel"""
        panel_height = 380
        
        # Glass-like panel
        cv2.rectangle(frame, (x, y), (x + width, y + panel_height), self.colors['slate_800'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + panel_height), self.colors['slate_700'], 2)
        
        # Header
        cv2.putText(frame, 'Financial Position', (x + 20, y + 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.85, self.colors['white'], 2, cv2.LINE_AA)
        
        # Divider line
        cv2.line(frame, (x + 20, y + 50), (x + width - 20, y + 50), self.colors['slate_700'], 1)
        
        stats_list = [
            ('Stability', self.stats.stability, self.colors['emerald_400']),
            ('Risk Exposure', self.stats.risk, self.colors['rose_400']),
            ('Liquidity', self.stats.liquidity, self.colors['blue_500']),
            ('Credit Rating', self.stats.credit_score, self.colors['blue_400'])
        ]
        
        stat_y = y + 80
        for label, value, color in stats_list:
            # Label
            cv2.putText(frame, label, (x + 20, stat_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.62, self.colors['gray_300'], 1, cv2.LINE_AA)
            
            # Value
            value_text = f'{int(value)}'
            cv2.putText(frame, value_text, (x + width - 65, stat_y + 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
            
            # Professional bar
            bar_y = stat_y + 12
            bar_width = width - 40
            bar_height = 10
            
            # Track
            cv2.rectangle(frame, (x + 20, bar_y), (x + 20 + bar_width, bar_y + bar_height), 
                         self.colors['slate_700'], -1)
            
            # Fill with gradient effect
            fill = int((value / 100.0) * bar_width)
            if fill > 0:
                cv2.rectangle(frame, (x + 20, bar_y), (x + 20 + fill, bar_y + bar_height), color, -1)
                
                # Subtle highlight
                cv2.line(frame, (x + 20, bar_y + 1), (x + 20 + fill, bar_y + 1), 
                        self.colors['white'], 1, cv2.LINE_AA)
            
            stat_y += 78
    
    def render_professional_context(self, frame: np.ndarray, x: int, y: int, width: int, text: str):
        """Professional context panel"""
        lines = self.wrap_text(text, 32)
        height = 30 + len(lines) * 30 + 20
        
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['slate_800'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['slate_700'], 2)
        
        text_y = y + 32
        for line in lines:
            cv2.putText(frame, line, (x + 20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                       self.colors['gray_100'], 1, cv2.LINE_AA)
            text_y += 30
    
    def render_professional_preview(self, frame: np.ndarray, x: int, y: int, width: int, decision: Decision):
        """Professional preview panel"""
        height = 300
        
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['slate_800'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['blue_500'], 3)
        
        cv2.putText(frame, 'Impact Analysis', (x + 22, y + 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.85, self.colors['white'], 2, cv2.LINE_AA)
        
        cv2.line(frame, (x + 20, y + 50), (x + width - 20, y + 50), self.colors['slate_700'], 1)
        
        impact_y = y + 85
        for stat, change in decision.impact.items():
            if change == 0:
                continue
            
            label = stat.replace('_', ' ').title()
            sign = '+' if change > 0 else ''
            color = self.colors['emerald_400'] if change > 0 else self.colors['rose_400']
            arrow = '↑' if change > 0 else '↓'
            
            cv2.putText(frame, f"{arrow} {label}", (x + 25, impact_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.62, self.colors['gray_300'], 1, cv2.LINE_AA)
            
            change_text = f'{sign}{int(change)}'
            cv2.putText(frame, change_text, (x + width - 75, impact_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
            
            impact_y += 48
    
    def render_professional_guidance(self, frame: np.ndarray, x: int, y: int, width: int):
        """Professional guidance panel"""
        height = 220
        
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['slate_800'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['slate_700'], 2)
        
        cv2.putText(frame, 'Instructions', (x + 22, y + 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.85, self.colors['white'], 2, cv2.LINE_AA)
        
        cv2.line(frame, (x + 20, y + 50), (x + width - 20, y + 50), self.colors['slate_700'], 1)
        
        tips = [
            'Hover over decision',
            'cards below to preview',
            'financial impact',
            '',
            'Hold for 3 seconds',
            'to confirm selection'
        ]
        
        tip_y = y + 85
        for tip in tips:
            cv2.putText(frame, tip, (x + 25, tip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                       self.colors['gray_300'], 1, cv2.LINE_AA)
            tip_y += 28
    
    def render_professional_card(self, frame: np.ndarray, x: int, y: int, w: int, h: int,
                                 decision: Decision, idx: int, is_hovered: bool, 
                                 hand_pos: Optional[Tuple[int, int]]):
        """Professional decision card"""
        # Background
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['slate_800'], -1)
        
        # Border
        border_color = self.colors['blue_500'] if is_hovered else self.colors['slate_700']
        border_thickness = 3 if is_hovered else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, border_thickness, cv2.LINE_AA)
        
        # Title
        title_lines = self.wrap_text(decision.title, 20)
        title_y = y + 35
        for line in title_lines:
            cv2.putText(frame, line, (x + 22, title_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.72, self.colors['white'], 2, cv2.LINE_AA)
            title_y += 32
        
        # Description
        desc_lines = self.wrap_text(decision.description, 28)
        desc_y = title_y + 8
        for line in desc_lines:
            cv2.putText(frame, line, (x + 22, desc_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['gray_300'], 1, cv2.LINE_AA)
            desc_y += 24
        
        # Progress bar
        if is_hovered and hand_pos:
            progress = min(1.0, (time.time() - self.hover_start_time) / self.hover_duration)
            
            bar_y = y + h - 28
            bar_width = w - 44
            bar_height = 10
            
            # Track
            cv2.rectangle(frame, (x + 22, bar_y), (x + 22 + bar_width, bar_y + bar_height), 
                         self.colors['slate_700'], -1)
            
            # Fill
            fill = int(progress * bar_width)
            if fill > 0:
                cv2.rectangle(frame, (x + 22, bar_y), (x + 22 + fill, bar_y + bar_height), 
                             self.colors['blue_500'], -1)
                
                # Highlight
                cv2.line(frame, (x + 22, bar_y + 1), (x + 22 + fill, bar_y + 1), 
                        self.colors['white'], 1, cv2.LINE_AA)
            
            # Percentage
            percent_text = f'{int(progress * 100)}%'
            cv2.putText(frame, percent_text, (x + w - 65, bar_y - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.colors['blue_500'], 2, cv2.LINE_AA)
    
    def render_fact_screen(self, frame: np.ndarray) -> np.ndarray:
        """Professional fact screen"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), self.colors['slate_900'], -1)
        frame = cv2.addWeighted(frame, 0.15, overlay, 0.85, 0)
        
        panel_width = 920
        panel_height = 320
        panel_x = (w - panel_width) // 2
        panel_y = (h - panel_height) // 2
        
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     self.colors['slate_800'], -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     self.colors['blue_500'], 4)
        
        # Icon
        icon_y = panel_y + 70
        cv2.circle(frame, (panel_x + panel_width // 2, icon_y), 35, self.colors['blue_500'], 3)
        cv2.putText(frame, 'i', (panel_x + panel_width // 2 - 10, icon_y + 14), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.4, self.colors['blue_500'], 3, cv2.LINE_AA)
        
        # Title
        cv2.putText(frame, 'Key Insight', (panel_x + panel_width//2 - 95, panel_y + 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['white'], 2, cv2.LINE_AA)
        
        # Fact
        fact_lines = self.wrap_text(self.current_fact, 65)
        fact_y = panel_y + 200
        for line in fact_lines:
            (text_w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
            cv2.putText(frame, line, (panel_x + (panel_width - text_w)//2, fact_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.62, self.colors['gray_100'], 1, cv2.LINE_AA)
            fact_y += 36
        
        if time.time() - self.fact_display_time > self.fact_duration:
            self.showing_fact = False
            self.phase += 1
            if self.phase >= len(self.phases):
                self.completed = True
        
        return frame
    
    def render_reflection(self, frame: np.ndarray) -> np.ndarray:
        """Professional completion screen"""
        h, w = frame.shape[:2]
        
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:] = self.colors['slate_900']
        frame = cv2.addWeighted(frame, 0.1, overlay, 0.9, 0)
        
        # Title
        title = 'Financial Simulation Complete'
        title_scale = 1.6
        (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 3)
        title_x = w//2 - title_w//2
        
        cv2.rectangle(frame, (title_x - 25, 35), (title_x + title_w + 25, 95), 
                     self.colors['slate_800'], -1)
        cv2.rectangle(frame, (title_x - 25, 35), (title_x + title_w + 25, 95), 
                     self.colors['blue_500'], 2)
        cv2.putText(frame, title, (title_x, 73),
                   cv2.FONT_HERSHEY_SIMPLEX, title_scale, self.colors['white'], 3, cv2.LINE_AA)
        
        # Stats
        self.render_professional_stats(frame, 50, 130, 350)
        
        # Analysis
        analysis_x = 460
        analysis_y = 130
        analysis_width = 770
        analysis_height = 400
        
        cv2.rectangle(frame, (analysis_x, analysis_y), (analysis_x + analysis_width, analysis_y + analysis_height), 
                     self.colors['slate_800'], -1)
        cv2.rectangle(frame, (analysis_x, analysis_y), (analysis_x + analysis_width, analysis_y + analysis_height), 
                     self.colors['slate_700'], 2)
        
        cv2.putText(frame, 'Strategy Assessment', (analysis_x + 30, analysis_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['white'], 2, cv2.LINE_AA)
        
        cv2.line(frame, (analysis_x + 30, analysis_y + 60), (analysis_x + analysis_width - 30, analysis_y + 60), 
                self.colors['slate_700'], 1)
        
        analysis = self.generate_analysis()
        text_y = analysis_y + 95
        for line in analysis:
            cv2.putText(frame, line, (analysis_x + 35, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.56, self.colors['gray_300'], 1, cv2.LINE_AA)
            text_y += 34
        
        # Achievement
        if (self.stats.stability >= 70 and self.stats.liquidity >= 30 and self.stats.risk < 60):
            badge_y = analysis_y + analysis_height - 60
            
            cv2.circle(frame, (analysis_x + analysis_width // 2, badge_y - 15), 40, 
                      self.colors['emerald_400'], 2)
            
            cv2.putText(frame, 'Balanced Strategy Achieved', (analysis_x + 210, badge_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['emerald_400'], 2, cv2.LINE_AA)
        
        # Restart
        button_width = 360
        button_height = 70
        button_x = w // 2 - button_width // 2
        button_y = h - 115
        
        cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + button_height), 
                     self.colors['slate_800'], -1)
        cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + button_height), 
                     self.colors['blue_500'], 3)
        
        cv2.putText(frame, 'Restart Simulation', (button_x + 65, button_y + 47),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.85, self.colors['white'], 2, cv2.LINE_AA)
        cv2.putText(frame, 'Press R', (button_x + 145, button_y - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['gray_300'], 1, cv2.LINE_AA)
        
        return frame
    
    def generate_analysis(self) -> List[str]:
        analysis = []
        
        if self.stats.risk > 70:
            analysis.append("High-Risk Profile: Strategy prioritized short-term gains over")
            analysis.append("long-term stability, increasing vulnerability to market volatility.")
        elif self.stats.risk < 40:
            analysis.append("Conservative Approach: Risk mitigation was central to strategy,")
            analysis.append("providing resilience but potentially limiting growth.")
        else:
            analysis.append("Balanced Risk Management: Strategy achieved equilibrium between")
            analysis.append("security and opportunity for sustainable wealth building.")
        
        analysis.append("")
        
        if self.stats.liquidity > 60:
            analysis.append("Strong Liquidity Position: Excellent cash reserves provide")
            analysis.append("optionality and protection against financial shocks.")
        elif self.stats.liquidity < 30:
            analysis.append("Liquidity Constraints: Limited reserves increase vulnerability")
            analysis.append("and reduce strategic flexibility in crisis scenarios.")
        
        analysis.append("")
        
        if self.insurance_level >= 1:
            analysis.append("Risk Transfer Implemented: Insurance coverage demonstrates")
            analysis.append("mature financial planning and catastrophic loss protection.")
        
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
        button_y = h - 165
        card_width = 340
        card_spacing = 50
        total_width = len(decisions) * card_width + (len(decisions) - 1) * card_spacing
        start_x = (w - total_width) // 2
        
        current_hover = -1
        for i in range(len(decisions)):
            card_x = start_x + i * (card_width + card_spacing)
            if self.check_hover(hand_pos, card_x, button_y, card_width, 140):
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
    print("Finance Module - Professional High Visibility Edition")
    print("• Sophisticated slate/blue color palette")
    print("• 3 second selection time")
    print("• Maximum readability")
    print("• Production-grade design")