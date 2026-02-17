"""
CYBERSEC AWARENESS SIMULATOR
Professional OpenCV-based interactive decision simulation
Hospital-grade dashboard quality — Personal Cybersecurity Education Edition
"""

import cv2
import numpy as np
import random
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum, auto


# ─────────────────────────────────────────────────────────────
#  COLOR PALETTE
# ─────────────────────────────────────────────────────────────

COLORS = {
    'bg_primary':       (20,  25,  35),
    'bg_secondary':     (30,  35,  50),
    'bg_panel':         (18,  22,  32),
    'accent_primary':   (255, 170,  60),
    'accent_secondary': (120, 200, 255),
    'accent_alert':     ( 60,  60, 220),
    'text_primary':     (240, 245, 255),
    'text_secondary':   (170, 190, 220),
    'text_dim':         (120, 140, 170),
    'hover_overlay':    ( 70, 110, 160),
    'stat_excellent':   (120, 220, 100),
    'stat_good':        ( 90, 200, 120),
    'stat_warning':     ( 80, 180, 220),
    'stat_critical':    ( 70,  70, 220),
}

W, H = 1280, 720
FPS  = 60
WINDOW = "CYBERSEC AWARENESS SIMULATOR"


# ─────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────

def alpha_blend(canvas: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(canvas, 1.0 - alpha, overlay, alpha, 0)


def wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text(canvas, text: str, pos: Tuple[int,int], color, font_scale=0.55,
              thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(canvas, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)


def draw_bold_text(canvas, text: str, pos: Tuple[int,int], color, font_scale=0.65, thickness=2):
    cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_panel(canvas, x, y, w, h, bg_color, border_color=None, border_thickness=1, corner_radius=4):
    sub = canvas[y:y+h, x:x+w]
    bg  = np.full_like(sub, bg_color[::-1])   # BGR
    alpha_blend_region(canvas, x, y, w, h, bg_color, 0.92)
    if border_color:
        cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), border_color[::-1], border_thickness)


def alpha_blend_region(canvas, x, y, w, h, color, alpha):
    roi = canvas[y:y+h, x:x+w].astype(np.float32)
    overlay = np.full_like(roi, [color[2], color[1], color[0]], dtype=np.float32)
    blended = cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0)
    canvas[y:y+h, x:x+w] = blended.astype(np.uint8)


def draw_stat_bar(canvas, x, y, width, height, value: float, label: str,
                  show_value: bool = True):
    """Draws a professional stat bar with label, value, and color threshold."""
    # Background track
    cv2.rectangle(canvas, (x, y), (x+width, y+height),
                  COLORS['bg_secondary'][::-1], -1)
    cv2.rectangle(canvas, (x, y), (x+width, y+height),
                  COLORS['text_dim'][::-1], 1)

    # Filled portion
    fill_w = int(width * max(0.0, min(1.0, value / 100.0)))

    # Dynamic color
    if value >= 75:
        bar_color = COLORS['stat_excellent']
    elif value >= 50:
        bar_color = COLORS['stat_good']
    elif value >= 30:
        bar_color = COLORS['stat_warning']
    else:
        bar_color = COLORS['stat_critical']

    if fill_w > 0:
        cv2.rectangle(canvas, (x, y), (x+fill_w, y+height),
                      bar_color[::-1], -1)
        # Shine effect
        shine_h = max(1, height // 3)
        shine_color = tuple(min(255, int(c*1.4)) for c in bar_color)
        cv2.rectangle(canvas, (x, y), (x+fill_w, y+shine_h),
                      shine_color[::-1], -1)

    # Label
    draw_text(canvas, label, (x, y - 7), COLORS['text_secondary'], 0.45)
    if show_value:
        val_str = f"{int(value)}"
        draw_text(canvas, val_str, (x + width + 5, y + height - 2),
                  COLORS['text_secondary'], 0.45)


def point_in_rect(px, py, rx, ry, rw, rh) -> bool:
    return rx <= px < rx+rw and ry <= py < ry+rh


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ─────────────────────────────────────────────────────────────
#  DATACLASS: CYBERSECURITY STATS
# ─────────────────────────────────────────────────────────────

@dataclass
class CyberSecurityStats:
    account_security:    float = 50.0
    data_privacy:        float = 50.0
    device_integrity:    float = 50.0
    awareness_level:     float = 50.0
    digital_reputation:  float = 50.0

    # Smoothed display values
    _display_acct:  float = field(default=50.0, repr=False)
    _display_priv:  float = field(default=50.0, repr=False)
    _display_dev:   float = field(default=50.0, repr=False)
    _display_aware: float = field(default=50.0, repr=False)
    _display_rep:   float = field(default=50.0, repr=False)

    @staticmethod
    def clamp(val: float) -> float:
        return max(0.0, min(100.0, val))

    def smooth_update(self, speed: float = 0.06):
        self._display_acct  = lerp(self._display_acct,  self.account_security,   speed)
        self._display_priv  = lerp(self._display_priv,  self.data_privacy,        speed)
        self._display_dev   = lerp(self._display_dev,   self.device_integrity,    speed)
        self._display_aware = lerp(self._display_aware, self.awareness_level,     speed)
        self._display_rep   = lerp(self._display_rep,   self.digital_reputation,  speed)

    def apply_delta(self, acct=0, priv=0, dev=0, aware=0, rep=0):
        self.account_security   = self.clamp(self.account_security   + acct)
        self.data_privacy       = self.clamp(self.data_privacy       + priv)
        self.device_integrity   = self.clamp(self.device_integrity   + dev)
        self.awareness_level    = self.clamp(self.awareness_level    + aware)
        self.digital_reputation = self.clamp(self.digital_reputation + rep)

    def average_score(self) -> float:
        return (self.account_security + self.data_privacy +
                self.device_integrity + self.awareness_level +
                self.digital_reputation) / 5.0

    @property
    def display_values(self):
        return [self._display_acct, self._display_priv,
                self._display_dev,  self._display_aware, self._display_rep]

    def sync_display(self):
        """Snap display to actual (used after fact screens)."""
        self._display_acct  = self.account_security
        self._display_priv  = self.data_privacy
        self._display_dev   = self.device_integrity
        self._display_aware = self.awareness_level
        self._display_rep   = self.digital_reputation


# ─────────────────────────────────────────────────────────────
#  IMAGE MANAGER
# ─────────────────────────────────────────────────────────────

class ImageManager:
    _cache: Dict[str, np.ndarray] = {}
    IMG_W, IMG_H = 480, 200

    PLACEHOLDER_SYMBOLS = {
        'phishing_email':   'PHISH',
        'social_media_risk':'SOCIAL',
        'public_wifi':      'WIFI',
        'ransomware_attack':'RANSOM',
        'identity_theft':   'ID-THEFT',
        'data_breach':      'BREACH',
        'digital_hygiene':  'HYGIENE',
    }

    @classmethod
    def get(cls, key: str) -> np.ndarray:
        if key in cls._cache:
            return cls._cache[key]

        # Try loading from disk
        for ext in ['.png', '.jpg', '.jpeg']:
            path = f"assets/{key}{ext}"
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, (cls.IMG_W, cls.IMG_H))
                    cls._cache[key] = img
                    return img

        # Generate placeholder
        img = cls._generate_placeholder(key)
        cls._cache[key] = img
        return img

    @classmethod
    def _generate_placeholder(cls, key: str) -> np.ndarray:
        img = np.zeros((cls.IMG_H, cls.IMG_W, 3), dtype=np.uint8)

        # Gradient background
        for row in range(cls.IMG_H):
            t = row / cls.IMG_H
            r = int(lerp(18, 40, t))
            g = int(lerp(22, 55, t))
            b = int(lerp(32, 80, t))
            img[row, :] = [b, g, r]

        # Grid lines (cyber aesthetic)
        for gx in range(0, cls.IMG_W, 40):
            cv2.line(img, (gx, 0), (gx, cls.IMG_H), (40, 60, 90), 1)
        for gy in range(0, cls.IMG_H, 20):
            cv2.line(img, (0, gy), (cls.IMG_W, gy), (40, 60, 90), 1)

        cx, cy = cls.IMG_W // 2, cls.IMG_H // 2

        # Draw icon based on key
        if 'phishing' in key:
            # Email envelope
            pts = np.array([[cx-60, cy-25],[cx+60, cy-25],
                             [cx+60, cy+25],[cx-60, cy+25]], np.int32)
            cv2.polylines(img, [pts], True, (255, 170, 60), 2)
            cv2.line(img, (cx-60, cy-25), (cx, cy+5), (255, 170, 60), 2)
            cv2.line(img, (cx+60, cy-25), (cx, cy+5), (255, 170, 60), 2)
            # Alert X
            cv2.line(img, (cx+40, cy-40), (cx+60, cy-20), (60, 60, 220), 2)
            cv2.line(img, (cx+60, cy-40), (cx+40, cy-20), (60, 60, 220), 2)

        elif 'social' in key:
            # People icons
            for ox in [-45, 0, 45]:
                cv2.circle(img, (cx+ox, cy-15), 12, (120, 200, 255), 2)
                cv2.ellipse(img, (cx+ox, cy+20), (16,12), 0, 180, 360,
                            (120, 200, 255), 2)

        elif 'wifi' in key:
            # WiFi arcs
            for r in [15, 30, 45]:
                cv2.ellipse(img, (cx, cy+20), (r, r), 0, 200, 340,
                            (120, 200, 255), 2)
            cv2.circle(img, (cx, cy+20), 4, (120, 200, 255), -1)
            # Warning !
            draw_bold_text(img, "?", (cx+30, cy-10), (255, 170, 60), 1.2, 3)

        elif 'ransomware' in key:
            # Lock icon
            cv2.rectangle(img, (cx-25, cy-10), (cx+25, cy+30),
                          (60, 60, 220), -1)
            cv2.rectangle(img, (cx-25, cy-10), (cx+25, cy+30),
                          (255, 170, 60), 2)
            cv2.ellipse(img, (cx, cy-20), (18, 18), 0, 180, 360,
                        (255, 170, 60), 2)
            draw_bold_text(img, "$", (cx-8, cy+20), (240, 245, 255), 0.8, 2)

        elif 'identity' in key:
            # ID card
            cv2.rectangle(img, (cx-55, cy-30), (cx+55, cy+30),
                          (30, 35, 50), -1)
            cv2.rectangle(img, (cx-55, cy-30), (cx+55, cy+30),
                          (255, 170, 60), 2)
            cv2.circle(img, (cx-30, cy), 16, (120, 200, 255), 2)
            for lx in range(cx, cx+45, 12):
                cv2.line(img, (lx, cy-12), (lx+8, cy-12),
                         (170, 190, 220), 1)
                cv2.line(img, (lx, cy-4), (lx+8, cy-4),
                         (120, 140, 170), 1)
            # Danger X
            draw_bold_text(img, "!", (cx+40, cy-40), (60, 60, 220), 1.0, 3)

        elif 'breach' in key:
            # Shield with crack
            pts = np.array([[cx, cy-40],[cx+35, cy-20],
                             [cx+35, cy+15],[cx, cy+40],
                             [cx-35, cy+15],[cx-35, cy-20]], np.int32)
            cv2.polylines(img, [pts], True, (60, 60, 220), 2)
            cv2.line(img, (cx, cy-30), (cx-10, cy+5), (255, 170, 60), 3)
            cv2.line(img, (cx-10, cy+5), (cx+5, cy+5), (255, 170, 60), 3)
            cv2.line(img, (cx+5, cy+5), (cx, cy+30), (255, 170, 60), 3)

        elif 'hygiene' in key:
            # Shield with checkmark
            pts = np.array([[cx, cy-40],[cx+35, cy-20],
                             [cx+35, cy+15],[cx, cy+40],
                             [cx-35, cy+15],[cx-35, cy-20]], np.int32)
            cv2.fillPoly(img, [pts], (25, 40, 25))
            cv2.polylines(img, [pts], True, (120, 220, 100), 2)
            cv2.line(img, (cx-15, cy+5), (cx, cy+20), (120, 220, 100), 3)
            cv2.line(img, (cx, cy+20), (cx+20, cy-10), (120, 220, 100), 3)

        # Label
        label = cls.PLACEHOLDER_SYMBOLS.get(key, key.upper())
        txt_w = len(label) * 9
        draw_text(img, label, (cx - txt_w//2, cls.IMG_H - 15),
                  COLORS['text_dim'], 0.5, 1)

        # Border
        cv2.rectangle(img, (0, 0), (cls.IMG_W-1, cls.IMG_H-1),
                      COLORS['accent_primary'][::-1], 1)

        return img


# ─────────────────────────────────────────────────────────────
#  DATA: PHASES & SCENARIOS
# ─────────────────────────────────────────────────────────────

class Phase(Enum):
    SCENARIO   = auto()
    FACT       = auto()
    CRISIS     = auto()
    CRISIS_FACT= auto()
    FINAL      = auto()

STAT_LABELS = [
    "Account Security",
    "Data Privacy",
    "Device Integrity",
    "Awareness Level",
    "Digital Reputation",
]

SCENARIOS = [
    # ── PHASE 1: Phishing ──
    {
        "phase_name":  "PHASE 1 — PHISHING ATTACK",
        "image_key":   "phishing_email",
        "context": (
            "You receive an urgent email from 'support@paypa1-secure.com' claiming "
            "your account will be locked in 24 hours. The email asks you to click a "
            "link and re-enter your credentials. The sender name shows 'PayPal Support' "
            "but the domain looks unusual. What do you do?"
        ),
        "decisions": [
            {
                "title":   "Click Link Immediately",
                "detail":  "Follow the email link and log in to 'fix' the problem.",
                "impacts": {"account_security": -30, "awareness_level": -20,
                            "digital_reputation": -10},
            },
            {
                "title":   "Inspect & Report Phishing",
                "detail":  "Check the sender domain, report the email, delete it.",
                "impacts": {"account_security": +25, "awareness_level": +30,
                            "digital_reputation": +10},
            },
            {
                "title":   "Ignore & Do Nothing",
                "detail":  "Don't click but also don't report — just delete it.",
                "impacts": {"account_security": -5, "awareness_level": +5,
                            "digital_reputation": 0},
            },
        ],
        "fact": (
            "FACT: 97% of people cannot identify phishing emails with certainty. "
            "Attackers use domain spoofing (paypa1 vs paypal) to deceive victims. "
            "Always verify the sender's full domain, hover over links before clicking, "
            "and report phishing to your email provider. A 2023 Verizon report found "
            "that 36% of all data breaches involved phishing."
        ),
    },

    # ── PHASE 2: Social Media ──
    {
        "phase_name":  "PHASE 2 — SOCIAL MEDIA PRIVACY RISK",
        "image_key":   "social_media_risk",
        "context": (
            "Your social media profile is fully public. It shows your full name, city, "
            "employer, and daily routines. A mutual contact recently shared a post "
            "revealing you're on vacation for 2 weeks. Strangers can see your phone "
            "number in the 'About' section. What action do you take?"
        ),
        "decisions": [
            {
                "title":   "Privatize & Audit Permissions",
                "detail":  "Set to private, review app permissions, remove personal data.",
                "impacts": {"data_privacy": +30, "digital_reputation": +15,
                            "awareness_level": +15},
            },
            {
                "title":   "Leave Public for Networking",
                "detail":  "Keep public visibility — the benefits outweigh the risks.",
                "impacts": {"data_privacy": -25, "digital_reputation": -10,
                            "awareness_level": -10},
            },
            {
                "title":   "Remove Suspicious Followers Only",
                "detail":  "Manually remove a few suspicious accounts but leave rest.",
                "impacts": {"data_privacy": +5, "digital_reputation": 0,
                            "awareness_level": +5},
            },
        ],
        "fact": (
            "FACT: Over 15 million identity theft cases occur annually in the US alone. "
            "Oversharing location data, daily routines, and personal identifiers on "
            "social media is the #1 enabler of social engineering attacks. Even 'private' "
            "networks can have compromised accounts. Conduct a privacy audit every 90 days. "
            "Remove all contact information that isn't strictly necessary."
        ),
    },

    # ── PHASE 3: Public WiFi ──
    {
        "phase_name":  "PHASE 3 — PUBLIC WIFI THREAT",
        "image_key":   "public_wifi",
        "context": (
            "You're at an international airport. You need to check your bank balance "
            "and respond to work emails. The airport offers free WiFi: 'AirportFreeWifi'. "
            "Your mobile data plan is running low. You notice a second network called "
            "'FREE_AIRPORT_WIFI' with a stronger signal. What do you do?"
        ),
        "decisions": [
            {
                "title":   "Connect with VPN Active",
                "detail":  "Use the official WiFi but tunnel all traffic via your VPN.",
                "impacts": {"device_integrity": +25, "account_security": +20,
                            "awareness_level": +20},
            },
            {
                "title":   "Connect Without Protection",
                "detail":  "Join the stronger signal network — probably fine.",
                "impacts": {"device_integrity": -30, "account_security": -25,
                            "data_privacy": -20},
            },
            {
                "title":   "Use Mobile Hotspot Instead",
                "detail":  "Create your own hotspot using mobile data only.",
                "impacts": {"device_integrity": +20, "account_security": +15,
                            "data_privacy": +10},
            },
        ],
        "fact": (
            "FACT: Man-in-the-Middle (MITM) attacks on public WiFi intercept data between "
            "you and the internet. 'Evil twin' rogue hotspots (the stronger signal network) "
            "are common in airports and cafes. Attackers can capture credentials, session "
            "cookies, and sensitive documents in seconds. A trusted VPN encrypts your "
            "tunnel end-to-end, making MITM attacks effectively useless. Never conduct "
            "financial transactions on unprotected public networks."
        ),
    },

    # ── PHASE 5: Digital Hygiene (after crisis) ──
    {
        "phase_name":  "PHASE 5 — LONG-TERM DIGITAL HYGIENE",
        "image_key":   "digital_hygiene",
        "context": (
            "After the security incident, you realize your digital habits need a systematic "
            "overhaul. You're evaluating three long-term strategies to harden your digital "
            "life. Each requires time investment but offers different levels of protection "
            "against future threats. Which strategy do you commit to?"
        ),
        "decisions": [
            {
                "title":   "Enable MFA Everywhere",
                "detail":  "Activate multi-factor authentication on all accounts immediately.",
                "impacts": {"account_security": +35, "awareness_level": +20,
                            "digital_reputation": +15},
            },
            {
                "title":   "Deploy Password Manager",
                "detail":  "Set up a password manager with unique 20+ char passwords for each site.",
                "impacts": {"account_security": +20, "data_privacy": +20,
                            "awareness_level": +15},
            },
            {
                "title":   "Ignore Long-term Improvements",
                "detail":  "Things have been fine so far — no immediate action needed.",
                "impacts": {"account_security": -20, "awareness_level": -15,
                            "digital_reputation": -15},
            },
        ],
        "fact": (
            "FACT: Microsoft research confirms that Multi-Factor Authentication (MFA) "
            "blocks 99.9% of automated account-takeover attacks. Yet only 57% of users "
            "have enabled it on their primary email. Password reuse across sites affects "
            "65% of people — a single breach can cascade across dozens of accounts. "
            "A password manager generating unique credentials per site is your single "
            "highest-ROI security investment. Both MFA and a password manager together "
            "eliminate over 95% of common attack vectors."
        ),
    },
]

CRISIS_EVENTS = [
    {
        "name":    "RANSOMWARE INFECTION",
        "image_key": "ransomware_attack",
        "description": (
            "CRITICAL ALERT: A ransomware payload has been detected on your device. "
            "Encrypted files are being held for $2,400 in cryptocurrency. System access "
            "is restricted. Backup integrity is unknown."
        ),
        "base_impacts": {"device_integrity": -40, "data_privacy": -25,
                         "digital_reputation": -20, "account_security": -10},
        "mitigation_stat": "device_integrity",
        "fact": (
            "FACT: Global ransomware damage costs exceeded $30 billion in 2023. "
            "Paying the ransom does NOT guarantee file recovery — 42% of paying victims "
            "never recover their data. The best defenses are regular offline backups "
            "(3-2-1 rule), keeping OS patched, and never opening unexpected email attachments. "
            "Ransomware exploits unpatched vulnerabilities 60% of the time."
        ),
    },
    {
        "name":    "IDENTITY THEFT ALERT",
        "image_key": "identity_theft",
        "description": (
            "CRITICAL ALERT: Your personal information has been used to open 3 fraudulent "
            "credit accounts. Your SSN, DOB, and home address have been exposed in an "
            "underground market. Unauthorized charges are appearing on your accounts."
        ),
        "base_impacts": {"data_privacy": -35, "account_security": -30,
                         "digital_reputation": -25, "awareness_level": -10},
        "mitigation_stat": "data_privacy",
        "fact": (
            "FACT: Identity theft victims spend an average of 200 hours recovering their "
            "identity and clearing fraudulent records. The dark web hosts over 15 billion "
            "stolen credentials. Credit freezes are FREE in the US and prevent unauthorized "
            "account openings. Monitor your credit report weekly via free services and "
            "set up fraud alerts. Early detection reduces recovery time by 80%."
        ),
    },
    {
        "name":    "ACCOUNT DATA BREACH",
        "image_key": "data_breach",
        "description": (
            "CRITICAL ALERT: A major platform you use has confirmed a breach exposing "
            "182 million user records. Your email, hashed password, location history, "
            "and payment method metadata are confirmed leaked."
        ),
        "base_impacts": {"account_security": -30, "data_privacy": -30,
                         "awareness_level": -15, "digital_reputation": -15},
        "mitigation_stat": "account_security",
        "fact": (
            "FACT: Have I Been Pwned (haveibeenpwned.com) tracks over 12 billion breached "
            "accounts. When a breach occurs, change your password immediately on that site "
            "AND any site where you reused it — password reuse is how breaches cascade. "
            "Enable breach notification services. A leaked hashed password can be cracked "
            "in seconds if you used a common word — minimum 16-character random passphrases "
            "are effectively uncrackable."
        ),
    },
]

GRADE_DATA = {
    "CYBER SECURITY CHAMPION": {
        "min_score": 80,
        "color": (120, 220, 100),
        "analysis": (
            "Exceptional performance. You demonstrated sophisticated threat awareness, "
            "applied defense-in-depth principles, and made optimal decisions under pressure. "
            "Your digital hygiene practices meet enterprise security standards. Continue "
            "staying current with emerging threats and consider cybersecurity certifications "
            "to formalize your expertise."
        ),
    },
    "DIGITALLY RESPONSIBLE": {
        "min_score": 65,
        "color": (90, 200, 120),
        "analysis": (
            "Solid security posture with room for targeted improvements. You identified "
            "most major threats correctly but some decision gaps exist. Focus on enabling "
            "MFA across all accounts, auditing social media privacy settings quarterly, "
            "and using a VPN consistently on public networks to close your remaining "
            "vulnerability windows."
        ),
    },
    "AT RISK USER": {
        "min_score": 50,
        "color": (80, 180, 220),
        "analysis": (
            "Significant security gaps were identified in your decision patterns. Several "
            "choices exposed you to preventable threats. Immediate action items: enable "
            "MFA on email and banking, install a reputable password manager, set all "
            "social media to private, and install a trusted VPN. These four steps will "
            "dramatically reduce your attack surface within 24 hours."
        ),
    },
    "CRITICAL VULNERABILITY": {
        "min_score": 0,
        "color": (70, 70, 220),
        "analysis": (
            "URGENT: Your current digital habits create severe exposure to identity theft, "
            "financial fraud, and data loss. Multiple high-risk choices were recorded. "
            "Take immediate action: change all passwords to unique 16+ character strings, "
            "enable MFA everywhere, run a full antivirus scan, freeze your credit, and "
            "contact your bank to review recent transactions. Consider a professional "
            "security audit of your accounts."
        ),
    },
}


# ─────────────────────────────────────────────────────────────
#  DECISION CARD
# ─────────────────────────────────────────────────────────────

@dataclass
class DecisionCard:
    index: int
    title: str
    detail: str
    impacts: dict
    x: int
    y: int
    w: int
    h: int
    hover_progress: float = 0.0
    is_hovered: bool = False

    def get_impact_lines(self) -> List[Tuple[str, bool]]:
        """Returns list of (label, is_positive) tuples."""
        STAT_MAP = {
            "account_security":   "Account Security",
            "data_privacy":       "Data Privacy",
            "device_integrity":   "Device Integrity",
            "awareness_level":    "Awareness Level",
            "digital_reputation": "Digital Reputation",
        }
        lines = []
        for key, delta in self.impacts.items():
            label = STAT_MAP.get(key, key)
            sign  = "+" if delta >= 0 else ""
            lines.append((f"{sign}{delta}  {label}", delta >= 0))
        return lines


# ─────────────────────────────────────────────────────────────
#  MAIN SIMULATOR
# ─────────────────────────────────────────────────────────────

class CyberSimulator:

    HOVER_CONFIRM_FRAMES = int(2.0 * FPS)
    FACT_DURATION_FRAMES = int(2.5 * FPS)

    # Layout constants
    LEFT_X,  LEFT_W  = 10,  240
    MID_X,   MID_W   = 260, 520
    RIGHT_X, RIGHT_W = 790, 230
    RIGHT_X2         = 1030
    RIGHT_W2         = 240

    HEADER_H  = 80
    COL_TOP   = 100
    COL_H     = 400
    DECK_Y    = 510
    DECK_H    = 195
    CARD_W    = 390
    CARD_H    = 180
    CARD_GAP  = 10

    def __init__(self):
        self.stats   = CyberSecurityStats()
        self.canvas  = np.zeros((H, W, 3), dtype=np.uint8)
        self.mouse_x = 0
        self.mouse_y = 0
        self.frame   = 0

        self.current_scenario_idx = 0
        self.phase                = Phase.SCENARIO
        self.decisions: List[DecisionCard] = []

        self.fact_text       = ""
        self.fact_timer      = 0
        self.fact_image_key  = ""

        self.crisis_event    = None
        self.crisis_fact_timer = 0

        self.hovered_card_idx: Optional[int] = None
        self.restart_hover   = 0.0
        self.restart_hovered = False

        self._build_scenario()

    # ── Setup ──────────────────────────────────────────────────

    def _build_scenario(self):
        if self.current_scenario_idx >= len(SCENARIOS):
            return
        sc   = SCENARIOS[self.current_scenario_idx]
        decs = sc["decisions"]

        total_w     = len(decs) * self.CARD_W + (len(decs)-1) * self.CARD_GAP
        start_x     = (W - total_w) // 2

        self.decisions = []
        for i, d in enumerate(decs):
            cx = start_x + i * (self.CARD_W + self.CARD_GAP)
            self.decisions.append(DecisionCard(
                index=i, title=d["title"], detail=d["detail"],
                impacts=d["impacts"],
                x=cx, y=self.DECK_Y, w=self.CARD_W, h=self.CARD_H,
            ))

    def _apply_decision(self, card: DecisionCard):
        self.stats.apply_delta(**card.impacts)
        sc = SCENARIOS[self.current_scenario_idx]
        self.fact_text      = sc["fact"]
        self.fact_image_key = sc["image_key"]
        self.fact_timer     = 0
        self.phase          = Phase.FACT

    def _advance_scenario(self):
        self.current_scenario_idx += 1

        # After phase 3 (index 2) → crisis event (Phase 4)
        if self.current_scenario_idx == 3:
            self._trigger_crisis()
            return

        # After crisis → phase 5 (index 3)
        if self.current_scenario_idx >= len(SCENARIOS):
            self.phase = Phase.FINAL
            return

        self._build_scenario()
        self.phase = Phase.SCENARIO

    def _trigger_crisis(self):
        self.crisis_event = random.choice(CRISIS_EVENTS)

        # Mitigation based on earlier stats
        mit_stat = self.crisis_event["mitigation_stat"]
        stat_val = getattr(self.stats, mit_stat)
        mit_factor = 1.0 - (stat_val / 100.0) * 0.65  # high stat → up to 65% less damage

        impacts = {}
        for k, v in self.crisis_event["base_impacts"].items():
            mitigated = int(v * mit_factor) if v < 0 else v
            impacts[k] = mitigated

        self.stats.apply_delta(**impacts)
        self.crisis_fact_timer = 0
        self.phase = Phase.CRISIS

    # ── Mouse callback ─────────────────────────────────────────

    def on_mouse(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y

    # ── Update ─────────────────────────────────────────────────

    def update(self):
        self.stats.smooth_update()
        self.frame += 1

        if self.phase == Phase.SCENARIO:
            self._update_scenario()

        elif self.phase == Phase.FACT:
            self.fact_timer += 1
            if self.fact_timer >= self.FACT_DURATION_FRAMES:
                self._advance_scenario()

        elif self.phase == Phase.CRISIS:
            self.crisis_fact_timer += 1
            if self.crisis_fact_timer >= int(1.5 * FPS):
                self.phase = Phase.CRISIS_FACT
                self.crisis_fact_timer = 0

        elif self.phase == Phase.CRISIS_FACT:
            self.crisis_fact_timer += 1
            if self.crisis_fact_timer >= self.FACT_DURATION_FRAMES:
                self.current_scenario_idx = 3  # Phase 5 scenario index
                self._build_scenario()
                self.phase = Phase.SCENARIO

        elif self.phase == Phase.FINAL:
            # Restart button hover
            rx, ry, rw, rh = W//2 - 120, H - 85, 240, 50
            if point_in_rect(self.mouse_x, self.mouse_y, rx, ry, rw, rh):
                self.restart_hover = min(1.0, self.restart_hover + 1/FPS)
                self.restart_hovered = True
                if self.restart_hover >= 1.0:
                    self._restart()
            else:
                self.restart_hover = max(0.0, self.restart_hover - 2/FPS)
                self.restart_hovered = False

    def _update_scenario(self):
        new_hovered = None
        for card in self.decisions:
            if point_in_rect(self.mouse_x, self.mouse_y,
                             card.x, card.y, card.w, card.h):
                card.is_hovered  = True
                card.hover_progress = min(1.0, card.hover_progress + 1 / self.HOVER_CONFIRM_FRAMES)
                new_hovered = card.index
                if card.hover_progress >= 1.0:
                    self._apply_decision(card)
                    return
            else:
                card.is_hovered     = False
                card.hover_progress = max(0.0, card.hover_progress - 2 / self.HOVER_CONFIRM_FRAMES)

        self.hovered_card_idx = new_hovered

    def _restart(self):
        self.stats             = CyberSecurityStats()
        self.current_scenario_idx = 0
        self.phase             = Phase.SCENARIO
        self.frame             = 0
        self.hovered_card_idx  = None
        self.restart_hover     = 0.0
        self.restart_hovered   = False
        self._build_scenario()

    # ── Render ─────────────────────────────────────────────────

    def render(self):
        self.canvas[:] = np.array(COLORS['bg_primary'][::-1], dtype=np.uint8)

        if self.phase in (Phase.SCENARIO, Phase.FACT):
            self._draw_header()
            self._draw_left_panel()
            self._draw_center_panel()
            self._draw_right_panel()
            if self.phase == Phase.SCENARIO:
                self._draw_decision_deck()
            else:
                self._draw_fact_overlay()

        elif self.phase == Phase.CRISIS:
            self._draw_header()
            self._draw_left_panel()
            self._draw_crisis_panel()

        elif self.phase == Phase.CRISIS_FACT:
            self._draw_header()
            self._draw_left_panel()
            self._draw_crisis_fact_overlay()

        elif self.phase == Phase.FINAL:
            self._draw_final_screen()

    # ── HEADER ─────────────────────────────────────────────────

    def _draw_header(self):
        # Background
        draw_panel(self.canvas, 0, 0, W, self.HEADER_H - 10,
                   COLORS['bg_secondary'],
                   border_color=COLORS['accent_primary'])

        # Title
        draw_bold_text(self.canvas, "CYBERSEC AWARENESS SIMULATOR",
                       (22, 38), COLORS['accent_primary'], 0.95, 2)

        # Subtitle / phase name
        if self.current_scenario_idx < len(SCENARIOS):
            phase_name = SCENARIOS[self.current_scenario_idx]["phase_name"]
        elif self.phase in (Phase.CRISIS, Phase.CRISIS_FACT):
            phase_name = "PHASE 4 — CYBER CRISIS EVENT"
        else:
            phase_name = "FINAL EVALUATION"

        draw_text(self.canvas, phase_name, (22, 62),
                  COLORS['accent_secondary'], 0.55, 1)

        # Accent line
        cv2.line(self.canvas,
                 (0, self.HEADER_H - 10),
                 (W, self.HEADER_H - 10),
                 COLORS['accent_primary'][::-1], 2)

        # Scanline effect
        t = self.frame * 0.04
        sx = int((math.sin(t) * 0.5 + 0.5) * W)
        cv2.line(self.canvas, (sx, 0), (sx, self.HEADER_H - 10),
                 (255, 220, 130, 30)[:3][::-1], 1)

        # Frame counter (subtle)
        draw_text(self.canvas, f"FRAME {self.frame:06d}",
                  (W - 180, 25), COLORS['text_dim'], 0.38, 1)

    # ── LEFT PANEL: Stat Bars ───────────────────────────────────

    def _draw_left_panel(self):
        px, py = self.LEFT_X, self.COL_TOP
        pw, ph = self.LEFT_W, self.COL_H
        draw_panel(self.canvas, px, py, pw, ph, COLORS['bg_panel'],
                   border_color=COLORS['accent_primary'])

        draw_bold_text(self.canvas, "SECURITY METRICS",
                       (px+12, py+24), COLORS['accent_primary'], 0.52, 1)
        cv2.line(self.canvas, (px+8, py+32), (px+pw-8, py+32),
                 COLORS['accent_primary'][::-1], 1)

        bar_w = pw - 50
        bar_h = 14
        bar_x = px + 16
        step  = (ph - 60) // 5

        display_vals = self.stats.display_values

        for i, (label, val) in enumerate(zip(STAT_LABELS, display_vals)):
            by = py + 55 + i * step
            draw_stat_bar(self.canvas, bar_x, by, bar_w, bar_h, val, label)

        # Score display
        score = self.stats.average_score()
        draw_text(self.canvas, f"OVERALL SCORE: {score:.1f}",
                  (px+16, py+ph-18), COLORS['text_secondary'], 0.45, 1)

        # Animated pulse on border
        pulse = 0.5 + 0.5 * math.sin(self.frame * 0.06)
        bc = tuple(int(c * pulse + COLORS['bg_panel'][j] * (1-pulse))
                   for j, c in enumerate(COLORS['accent_primary']))
        cv2.rectangle(self.canvas, (px, py), (px+pw-1, py+ph-1),
                      bc[::-1], 1)

    # ── CENTER PANEL ────────────────────────────────────────────

    def _draw_center_panel(self):
        px, py = self.MID_X, self.COL_TOP
        pw, ph = self.MID_W, self.COL_H

        draw_panel(self.canvas, px, py, pw, ph, COLORS['bg_panel'],
                   border_color=COLORS['accent_secondary'])

        if self.current_scenario_idx >= len(SCENARIOS):
            return

        sc = SCENARIOS[self.current_scenario_idx]

        # Section title
        draw_bold_text(self.canvas, "SCENARIO BRIEFING",
                       (px+12, py+24), COLORS['accent_secondary'], 0.52, 1)
        cv2.line(self.canvas, (px+8, py+32), (px+pw-8, py+32),
                 COLORS['accent_secondary'][::-1], 1)

        # Image area
        img_y = py + 42
        img_h = 200
        img   = ImageManager.get(sc["image_key"])
        img_resized = cv2.resize(img, (pw - 16, img_h))
        roi_y1 = img_y
        roi_y2 = img_y + img_h
        roi_x1 = px + 8
        roi_x2 = px + 8 + (pw - 16)
        self.canvas[roi_y1:roi_y2, roi_x1:roi_x2] = img_resized
        cv2.rectangle(self.canvas, (roi_x1, roi_y1), (roi_x2-1, roi_y2-1),
                      COLORS['accent_secondary'][::-1], 1)

        # Context text
        text_y = py + 260
        lines  = wrap_text(sc["context"], 58)
        for i, line in enumerate(lines[:6]):
            draw_text(self.canvas, line, (px+12, text_y + i*22),
                      COLORS['text_primary'], 0.47, 1)

        # Instruction
        draw_text(self.canvas, "HOVER A DECISION CARD FOR 2 SEC TO CONFIRM",
                  (px+12, py+ph-18), COLORS['text_dim'], 0.38, 1)

    # ── RIGHT PANEL: Decision Preview ──────────────────────────

    def _draw_right_panel(self):
        px, py = self.RIGHT_X, self.COL_TOP
        pw, ph = self.RIGHT_W, self.COL_H

        draw_panel(self.canvas, px, py, pw, ph, COLORS['bg_panel'],
                   border_color=COLORS['text_dim'])

        draw_bold_text(self.canvas, "IMPACT PREVIEW",
                       (px+12, py+24), COLORS['text_secondary'], 0.50, 1)
        cv2.line(self.canvas, (px+8, py+32), (px+pw-8, py+32),
                 COLORS['text_dim'][::-1], 1)

        if self.hovered_card_idx is None:
            draw_text(self.canvas, "Hover a decision card",
                      (px+12, py+65), COLORS['text_dim'], 0.45, 1)
            draw_text(self.canvas, "to preview its impact",
                      (px+12, py+85), COLORS['text_dim'], 0.45, 1)
            draw_text(self.canvas, "on your security stats.",
                      (px+12, py+105), COLORS['text_dim'], 0.45, 1)
            return

        card = self.decisions[self.hovered_card_idx]

        # Title
        title_lines = wrap_text(card.title, 26)
        for i, l in enumerate(title_lines):
            draw_bold_text(self.canvas, l, (px+12, py+55+i*22),
                           COLORS['accent_primary'], 0.52, 1)

        # Detail
        detail_lines = wrap_text(card.detail, 30)
        for i, l in enumerate(detail_lines[:3]):
            draw_text(self.canvas, l, (px+12, py+100+i*18),
                      COLORS['text_secondary'], 0.42, 1)

        # Divider
        cv2.line(self.canvas, (px+8, py+158), (px+pw-8, py+158),
                 COLORS['text_dim'][::-1], 1)

        draw_text(self.canvas, "STAT IMPACTS:", (px+12, py+175),
                  COLORS['text_dim'], 0.42, 1)

        impacts = card.get_impact_lines()
        for i, (txt, positive) in enumerate(impacts):
            color = COLORS['stat_excellent'] if positive else COLORS['stat_critical']
            draw_text(self.canvas, txt, (px+16, py+196+i*22),
                      color, 0.44, 1)

        # Right panel 2 — hover progress
        if self.hovered_card_idx is not None:
            c = self.decisions[self.hovered_card_idx]
            self._draw_hover_progress_mini(px, py+ph-50, pw)

    def _draw_hover_progress_mini(self, x, y, w):
        draw_text(self.canvas, "HOLD TO CONFIRM:", (x+12, y),
                  COLORS['text_dim'], 0.38, 1)
        bar_x, bar_y = x+8, y+10
        bar_w = w - 16
        if self.hovered_card_idx is not None:
            prog = self.decisions[self.hovered_card_idx].hover_progress
        else:
            prog = 0.0
        cv2.rectangle(self.canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+12),
                      COLORS['bg_secondary'][::-1], -1)
        fill = int(prog * bar_w)
        if fill > 0:
            col = COLORS['accent_primary'] if prog < 1.0 else COLORS['stat_excellent']
            cv2.rectangle(self.canvas, (bar_x, bar_y), (bar_x+fill, bar_y+12),
                          col[::-1], -1)
        cv2.rectangle(self.canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+12),
                      COLORS['text_dim'][::-1], 1)
        pct = f"{int(prog*100)}%"
        draw_text(self.canvas, pct, (bar_x+bar_w+6, bar_y+11),
                  COLORS['text_dim'], 0.38, 1)

    # ── DECISION DECK ───────────────────────────────────────────

    def _draw_decision_deck(self):
        # Panel background
        draw_panel(self.canvas, 0, self.DECK_Y - 10, W, H - self.DECK_Y + 10,
                   COLORS['bg_secondary'],
                   border_color=COLORS['text_dim'])

        draw_text(self.canvas, "SELECT YOUR RESPONSE STRATEGY",
                  (15, self.DECK_Y + 12), COLORS['accent_primary'], 0.48, 1)

        for card in self.decisions:
            self._draw_card(card)

    def _draw_card(self, card: DecisionCard):
        x, y, w, h = card.x, card.y, card.w, card.h

        # Shadow
        cv2.rectangle(self.canvas, (x+3, y+3), (x+w+3, y+h+3),
                      (10, 10, 20), -1)

        # Base
        draw_panel(self.canvas, x, y, w, h, COLORS['bg_panel'],
                   border_color=COLORS['accent_primary'])

        # Hover overlay
        if card.is_hovered and card.hover_progress > 0:
            overlay = self.canvas.copy()
            alpha   = card.hover_progress * 0.35
            alpha_blend_region(self.canvas, x+1, y+1, w-2, h-2,
                               COLORS['hover_overlay'], alpha)

        # Card number badge
        badge_color = COLORS['accent_primary']
        cv2.circle(self.canvas, (x+28, y+28), 14, badge_color[::-1], -1)
        draw_bold_text(self.canvas, str(card.index+1), (x+22, y+34),
                       COLORS['bg_primary'], 0.55, 2)

        # Title
        title_lines = wrap_text(card.title, 38)
        for i, l in enumerate(title_lines):
            draw_bold_text(self.canvas, l, (x+50, y+30+i*22),
                           COLORS['text_primary'], 0.58, 1)

        # Detail
        detail_lines = wrap_text(card.detail, 50)
        for i, l in enumerate(detail_lines[:2]):
            draw_text(self.canvas, l, (x+14, y+72+i*18),
                      COLORS['text_secondary'], 0.43, 1)

        # Impact summary
        cv2.line(self.canvas, (x+8, y+108), (x+w-8, y+108),
                 COLORS['text_dim'][::-1], 1)
        impacts = card.get_impact_lines()
        for i, (txt, positive) in enumerate(impacts[:3]):
            col = COLORS['stat_excellent'] if positive else COLORS['stat_critical']
            ix  = x + 14 + (i % 2) * (w // 2)
            iy  = y + 126 + (i // 2) * 18
            draw_text(self.canvas, txt, (ix, iy), col, 0.40, 1)

        # Hover confirm bar
        if card.is_hovered:
            by  = y + h - 20
            bx  = x + 8
            bw  = w - 16
            bh  = 10
            cv2.rectangle(self.canvas, (bx, by), (bx+bw, by+bh),
                          COLORS['bg_primary'][::-1], -1)
            fill = int(card.hover_progress * bw)
            if fill > 0:
                cv2.rectangle(self.canvas, (bx, by), (bx+fill, by+bh),
                              COLORS['accent_primary'][::-1], -1)
            cv2.rectangle(self.canvas, (bx, by), (bx+bw, by+bh),
                          COLORS['text_dim'][::-1], 1)

            # Confirm text blink
            if self.frame % 30 < 20:
                draw_text(self.canvas, "HOLD TO CONFIRM",
                          (x + w//2 - 60, y + h - 26),
                          COLORS['accent_primary'], 0.40, 1)

        # Border glow on hover
        border_col = COLORS['accent_secondary'] if card.is_hovered else COLORS['accent_primary']
        cv2.rectangle(self.canvas, (x, y), (x+w-1, y+h-1),
                      border_col[::-1], 2 if card.is_hovered else 1)

    # ── FACT OVERLAY ────────────────────────────────────────────

    def _draw_fact_overlay(self):
        # Semi-transparent overlay
        overlay = self.canvas.copy()
        alpha_blend_region(self.canvas, 0, 0, W, H, COLORS['bg_primary'], 0.88)

        # Fact panel
        fx, fy = 180, 100
        fw, fh = 920, 400

        draw_panel(self.canvas, fx, fy, fw, fh, COLORS['bg_panel'],
                   border_color=COLORS['accent_secondary'], border_thickness=2)

        draw_bold_text(self.canvas, "EDUCATIONAL INSIGHT",
                       (fx+20, fy+36), COLORS['accent_secondary'], 0.75, 2)
        cv2.line(self.canvas, (fx+12, fy+48), (fx+fw-12, fy+48),
                 COLORS['accent_secondary'][::-1], 2)

        # Fact image thumbnail
        img = ImageManager.get(self.fact_image_key)
        thumb = cv2.resize(img, (220, 130))
        roi_x1, roi_y1 = fx+fw-236, fy+60
        self.canvas[roi_y1:roi_y1+130, roi_x1:roi_x1+220] = thumb
        cv2.rectangle(self.canvas, (roi_x1, roi_y1),
                      (roi_x1+220, roi_y1+130),
                      COLORS['accent_secondary'][::-1], 1)

        # Fact text
        lines = wrap_text(self.fact_text, 70)
        for i, line in enumerate(lines[:10]):
            draw_text(self.canvas, line, (fx+20, fy+70+i*26),
                      COLORS['text_primary'], 0.50, 1)

        # Progress bar
        progress = min(1.0, self.fact_timer / self.FACT_DURATION_FRAMES)
        bx, by = fx+20, fy+fh-30
        bw = fw - 40
        cv2.rectangle(self.canvas, (bx, by), (bx+bw, by+14),
                      COLORS['bg_secondary'][::-1], -1)
        fill = int(progress * bw)
        cv2.rectangle(self.canvas, (bx, by), (bx+fill, by+14),
                      COLORS['accent_secondary'][::-1], -1)
        cv2.rectangle(self.canvas, (bx, by), (bx+bw, by+14),
                      COLORS['text_dim'][::-1], 1)
        draw_text(self.canvas, f"AUTO-ADVANCING... {int(progress*100)}%",
                  (bx+6, by+11), COLORS['bg_primary'], 0.40, 1)

    # ── CRISIS PANEL ────────────────────────────────────────────

    def _draw_crisis_panel(self):
        if not self.crisis_event:
            return

        # Alert border (pulsing red)
        pulse = int(128 + 127 * math.sin(self.frame * 0.15))
        cv2.rectangle(self.canvas, (0, 0), (W-1, H-1),
                      (0, 0, pulse), 3)

        cx_panel = 200
        cy_panel = 95
        cw_panel = 880
        ch_panel = 430

        draw_panel(self.canvas, cx_panel, cy_panel, cw_panel, ch_panel,
                   COLORS['bg_panel'],
                   border_color=COLORS['accent_alert'], border_thickness=3)

        # Alert header
        draw_bold_text(self.canvas, f"⚠  CYBER CRISIS: {self.crisis_event['name']}",
                       (cx_panel+20, cy_panel+38), COLORS['accent_alert'], 0.80, 2)
        cv2.line(self.canvas, (cx_panel+12, cy_panel+50),
                 (cx_panel+cw_panel-12, cy_panel+50),
                 COLORS['accent_alert'][::-1], 2)

        # Image
        img     = ImageManager.get(self.crisis_event['image_key'])
        thumb   = cv2.resize(img, (340, 180))
        img_rx  = cx_panel + cw_panel - 360
        img_ry  = cy_panel + 62
        self.canvas[img_ry:img_ry+180, img_rx:img_rx+340] = thumb
        cv2.rectangle(self.canvas, (img_rx, img_ry),
                      (img_rx+340, img_ry+180),
                      COLORS['accent_alert'][::-1], 2)

        # Description
        lines = wrap_text(self.crisis_event['description'], 62)
        for i, line in enumerate(lines[:5]):
            draw_text(self.canvas, line, (cx_panel+20, cy_panel+72+i*26),
                      COLORS['text_primary'], 0.50, 1)

        # Impact notice
        draw_text(self.canvas, "SECURITY METRICS IMPACTED — MITIGATION APPLIED BASED ON PRIOR DECISIONS",
                  (cx_panel+20, cy_panel+210), COLORS['stat_critical'], 0.44, 1)

        # Show all applied impacts
        for i, (k, v) in enumerate(self.crisis_event['base_impacts'].items()):
            label = STAT_LABELS[["account_security","data_privacy","device_integrity",
                                  "awareness_level","digital_reputation"].index(k)]
            sign  = "+" if v >= 0 else ""
            col   = COLORS['stat_excellent'] if v >= 0 else COLORS['stat_critical']
            draw_text(self.canvas, f"{sign}{v}  {label}",
                      (cx_panel + 20 + (i % 2) * 420, cy_panel + 235 + (i // 2) * 24),
                      col, 0.45, 1)

        # Countdown
        prog = min(1.0, self.crisis_fact_timer / int(1.5*FPS))
        draw_text(self.canvas, f"ASSESSING DAMAGE... {int(prog*100)}%",
                  (cx_panel+20, cy_panel+ch_panel-30),
                  COLORS['accent_alert'], 0.48, 1)

        # Mitigation note
        mit = self.crisis_event['mitigation_stat'].replace("_", " ").title()
        draw_text(self.canvas, f"MITIGATION FACTOR: High {mit} reduced damage",
                  (cx_panel+20, cy_panel+ch_panel-55),
                  COLORS['text_secondary'], 0.42, 1)

    # ── CRISIS FACT ─────────────────────────────────────────────

    def _draw_crisis_fact_overlay(self):
        if not self.crisis_event:
            return
        alpha_blend_region(self.canvas, 0, 0, W, H, COLORS['bg_primary'], 0.90)

        fx, fy = 150, 90
        fw, fh = 980, 430

        draw_panel(self.canvas, fx, fy, fw, fh, COLORS['bg_panel'],
                   border_color=COLORS['accent_alert'], border_thickness=2)

        draw_bold_text(self.canvas,
                       f"CRISIS EDUCATION — {self.crisis_event['name']}",
                       (fx+20, fy+36), COLORS['accent_alert'], 0.70, 2)
        cv2.line(self.canvas, (fx+12, fy+48), (fx+fw-12, fy+48),
                 COLORS['accent_alert'][::-1], 2)

        lines = wrap_text(self.crisis_event['fact'], 78)
        for i, line in enumerate(lines[:10]):
            draw_text(self.canvas, line, (fx+20, fy+70+i*28),
                      COLORS['text_primary'], 0.50, 1)

        progress = min(1.0, self.crisis_fact_timer / self.FACT_DURATION_FRAMES)
        bx, by   = fx+20, fy+fh-32
        bw       = fw - 40
        cv2.rectangle(self.canvas, (bx, by), (bx+bw, by+14),
                      COLORS['bg_secondary'][::-1], -1)
        fill = int(progress * bw)
        cv2.rectangle(self.canvas, (bx, by), (bx+fill, by+14),
                      COLORS['accent_alert'][::-1], -1)
        cv2.rectangle(self.canvas, (bx, by), (bx+bw, by+14),
                      COLORS['text_dim'][::-1], 1)
        draw_text(self.canvas, f"CONTINUING TO PHASE 5... {int(progress*100)}%",
                  (bx+6, by+11), COLORS['text_primary'], 0.40, 1)

    # ── FINAL SCREEN ────────────────────────────────────────────

    def _draw_final_screen(self):
        self.canvas[:] = np.array(COLORS['bg_primary'][::-1], dtype=np.uint8)

        score = self.stats.average_score()

        # Determine grade
        grade_key   = "CRITICAL VULNERABILITY"
        grade_info  = GRADE_DATA["CRITICAL VULNERABILITY"]
        for gname, gdata in GRADE_DATA.items():
            if score >= gdata["min_score"]:
                grade_key  = gname
                grade_info = gdata
                break

        grade_color = grade_info["color"]

        # Top banner
        cv2.rectangle(self.canvas, (0, 0), (W, 100),
                      grade_color[::-1], -1)
        draw_bold_text(self.canvas, "FINAL EVALUATION — CYBERSEC SIMULATION",
                       (22, 38), COLORS['bg_primary'], 0.85, 2)
        draw_bold_text(self.canvas, grade_key,
                       (22, 78), COLORS['bg_primary'], 0.75, 2)

        # Score
        draw_bold_text(self.canvas, f"OVERALL SCORE: {score:.1f} / 100",
                       (22, 132), COLORS['accent_primary'], 0.85, 2)

        # Analysis
        analysis_lines = wrap_text(grade_info["analysis"], 90)
        for i, line in enumerate(analysis_lines[:5]):
            draw_text(self.canvas, line, (22, 165+i*24),
                      COLORS['text_primary'], 0.50, 1)

        cv2.line(self.canvas, (12, 295), (W-12, 295),
                 COLORS['accent_primary'][::-1], 1)

        # Final stat bars
        draw_bold_text(self.canvas, "FINAL SECURITY PROFILE:",
                       (22, 322), COLORS['accent_secondary'], 0.60, 1)

        display_vals = self.stats.display_values
        bar_w = 500
        bar_h = 18

        for i, (label, val) in enumerate(zip(STAT_LABELS, display_vals)):
            bx = 22
            by = 340 + i * 38
            draw_text(self.canvas, label, (bx, by - 6),
                      COLORS['text_secondary'], 0.50, 1)
            draw_stat_bar(self.canvas, bx, by, bar_w, bar_h, val, "", show_value=False)
            draw_bold_text(self.canvas, f"{int(val)}",
                           (bx + bar_w + 12, by + bar_h),
                           COLORS['text_primary'], 0.52, 1)

        # Right column: radar-style breakdown
        self._draw_final_radar(700, 300)

        # Restart button
        rx, ry, rw, rh = W//2 - 120, H - 85, 240, 50
        prog = self.restart_hover

        cv2.rectangle(self.canvas, (rx, ry), (rx+rw, ry+rh),
                      COLORS['bg_secondary'][::-1], -1)
        if prog > 0:
            fill_col = COLORS['stat_excellent'] if prog < 1.0 else grade_color
            cv2.rectangle(self.canvas, (rx, ry),
                          (rx + int(rw * prog), ry + rh),
                          fill_col[::-1], -1)
        cv2.rectangle(self.canvas, (rx, ry), (rx+rw, ry+rh),
                      COLORS['accent_primary'][::-1], 2)

        label = "HOLD TO RESTART" if not self.restart_hovered or prog < 1.0 else "RESTARTING..."
        tw = len(label) * 9
        draw_bold_text(self.canvas, label,
                       (rx + rw//2 - tw//2, ry + 32),
                       COLORS['text_primary'], 0.58, 1)

    def _draw_final_radar(self, cx, cy):
        """Draw a simple hexagonal radar chart for final stats."""
        n       = 5
        radius  = 130
        display = self.stats.display_values

        # Background rings
        for r_pct in [0.25, 0.5, 0.75, 1.0]:
            pts = []
            for i in range(n):
                angle = math.pi/2 + 2*math.pi*i/n
                r     = radius * r_pct
                pts.append((int(cx + r * math.cos(angle)),
                             int(cy - r * math.sin(angle))))
            pts_np = np.array(pts, np.int32)
            cv2.polylines(self.canvas, [pts_np], True,
                          COLORS['text_dim'][::-1], 1)

        # Axes
        for i in range(n):
            angle = math.pi/2 + 2*math.pi*i/n
            ex    = int(cx + radius * math.cos(angle))
            ey    = int(cy - radius * math.sin(angle))
            cv2.line(self.canvas, (cx, cy), (ex, ey),
                     COLORS['text_dim'][::-1], 1)

        # Data polygon
        data_pts = []
        for i, val in enumerate(display):
            angle = math.pi/2 + 2*math.pi*i/n
            r     = radius * val / 100.0
            data_pts.append((int(cx + r * math.cos(angle)),
                             int(cy - r * math.sin(angle))))
        dp_np = np.array(data_pts, np.int32)

        # Fill
        overlay = self.canvas.copy()
        cv2.fillPoly(overlay, [dp_np], COLORS['accent_secondary'][::-1])
        self.canvas[:] = alpha_blend(self.canvas, overlay, 0.3)
        cv2.polylines(self.canvas, [dp_np], True,
                      COLORS['accent_secondary'][::-1], 2)

        # Point dots
        for pt in data_pts:
            cv2.circle(self.canvas, pt, 4,
                       COLORS['accent_primary'][::-1], -1)

        # Labels
        for i, label in enumerate(STAT_LABELS):
            angle = math.pi/2 + 2*math.pi*i/n
            lx    = int(cx + (radius + 22) * math.cos(angle))
            ly    = int(cy - (radius + 22) * math.sin(angle))
            words = label.split()
            for j, w in enumerate(words):
                draw_text(self.canvas, w, (lx - len(w)*4, ly + j*14),
                          COLORS['text_secondary'], 0.38, 1)

    # ── Main loop ───────────────────────────────────────────────

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, W, H)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        delay = max(1, int(1000 / FPS))

        print("=" * 60)
        print("  CYBERSEC AWARENESS SIMULATOR")
        print("  Resolution: 1280x720 @ 60 FPS")
        print("  Hover decision cards for 2 seconds to confirm.")
        print("  Press ESC or Q to quit.")
        print("=" * 60)

        while True:
            self.update()
            self.render()
            cv2.imshow(WINDOW, self.canvas)

            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

        cv2.destroyAllWindows()
        print("Session ended. Stay cyber-safe!")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sim = CyberSimulator()
    sim.run()