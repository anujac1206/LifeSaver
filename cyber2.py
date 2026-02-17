"""
CYBERSEC AWARENESS SIMULATOR  v2.0
Fixes: crash on phase advance, em-dash encoding, card overflow,
       overlapping text, cleaner layout with breathing room.
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
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
W, H   = 1280, 720
FPS    = 60
WINDOW = "CYBERSEC AWARENESS SIMULATOR"

COLORS = {
    'bg_primary':       (20,  25,  35),
    'bg_secondary':     (28,  33,  48),
    'bg_panel':         (15,  19,  28),
    'accent_primary':   (255, 170,  60),
    'accent_secondary': (100, 190, 255),
    'accent_alert':     ( 60,  60, 220),
    'text_primary':     (235, 242, 255),
    'text_secondary':   (160, 180, 215),
    'text_dim':         (100, 120, 155),
    'hover_overlay':    ( 55,  95, 145),
    'stat_excellent':   (100, 210,  85),
    'stat_good':        ( 75, 185, 105),
    'stat_warning':     ( 75, 170, 215),
    'stat_critical':    ( 60,  60, 215),
}

STAT_LABELS = [
    "Account Security",
    "Data Privacy",
    "Device Integrity",
    "Awareness Level",
    "Digital Reputation",
]

# Layout - fixed non-overlapping zones
HEADER_H  = 75
COL_Y     = 84
COL_H     = 410
DECK_Y    = 502
DECK_H    = H - DECK_Y   # = 218

LEFT_X, LEFT_W   = 8,   230
MID_X,  MID_W    = 246, 530
RIGHT_X, RIGHT_W = 784, 242

CARD_W, CARD_H = 395, 205
CARD_GAP       = 12

# ─────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────

def lerp(a, b, t):
    return a + (b - a) * t

def clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))

def point_in_rect(px, py, rx, ry, rw, rh):
    return rx <= px < rx + rw and ry <= py < ry + rh

def bgr(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

def wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if len(test) <= max_chars:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def put(canvas, text, pos, color, scale=0.50, thick=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(canvas, str(text), pos, font, scale, bgr(color), thick, cv2.LINE_AA)

def put_bold(canvas, text, pos, color, scale=0.60, thick=2):
    cv2.putText(canvas, str(text), pos, cv2.FONT_HERSHEY_DUPLEX, scale, bgr(color), thick, cv2.LINE_AA)

def fill_rect(canvas, x, y, w, h, color, alpha=1.0):
    if w <= 0 or h <= 0:
        return
    x, y, w, h = int(x), int(y), int(w), int(h)
    x2 = min(x+w, canvas.shape[1])
    y2 = min(y+h, canvas.shape[0])
    if x2 <= x or y2 <= y:
        return
    if alpha >= 0.999:
        canvas[y:y2, x:x2] = bgr(color)
    else:
        roi = canvas[y:y2, x:x2].astype(np.float32)
        ov  = np.full_like(roi, bgr(color), dtype=np.float32)
        canvas[y:y2, x:x2] = cv2.addWeighted(roi, 1-alpha, ov, alpha, 0).astype(np.uint8)

def border_rect(canvas, x, y, w, h, color, thick=1):
    cv2.rectangle(canvas, (int(x), int(y)), (int(x+w-1), int(y+h-1)), bgr(color), thick)

def panel(canvas, x, y, w, h, bg, border=None, bthick=1, alpha=0.96):
    fill_rect(canvas, x, y, w, h, bg, alpha)
    if border:
        border_rect(canvas, x, y, w, h, border, bthick)

def hline(canvas, x1, x2, y, color, thick=1):
    cv2.line(canvas, (int(x1), int(y)), (int(x2), int(y)), bgr(color), thick)

def stat_bar(canvas, x, y, bw, bh, val):
    fill_rect(canvas, x, y, bw, bh, COLORS['bg_secondary'])
    border_rect(canvas, x, y, bw, bh, COLORS['text_dim'])
    fw = int(bw * clamp(val) / 100)
    if fw > 0:
        if val >= 75:   col = COLORS['stat_excellent']
        elif val >= 50: col = COLORS['stat_good']
        elif val >= 30: col = COLORS['stat_warning']
        else:           col = COLORS['stat_critical']
        fill_rect(canvas, x, y, fw, bh, col)
        sh = max(1, bh // 3)
        sc = tuple(min(255, int(c * 1.35)) for c in col)
        fill_rect(canvas, x, y, fw, sh, sc)

def progress_bar(canvas, x, y, bw, bh, prog, color):
    fill_rect(canvas, x, y, bw, bh, COLORS['bg_secondary'])
    fw = int(bw * clamp(prog, 0, 1))
    if fw > 0:
        fill_rect(canvas, x, y, fw, bh, color)
    border_rect(canvas, x, y, bw, bh, COLORS['text_dim'])


# ─────────────────────────────────────────────────────────────
#  STATS DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class Stats:
    account_security:   float = 50.0
    data_privacy:       float = 50.0
    device_integrity:   float = 50.0
    awareness_level:    float = 50.0
    digital_reputation: float = 50.0

    _d_acct:  float = field(default=50.0, repr=False)
    _d_priv:  float = field(default=50.0, repr=False)
    _d_dev:   float = field(default=50.0, repr=False)
    _d_aware: float = field(default=50.0, repr=False)
    _d_rep:   float = field(default=50.0, repr=False)

    def smooth(self, spd=0.07):
        self._d_acct  = lerp(self._d_acct,  self.account_security,   spd)
        self._d_priv  = lerp(self._d_priv,  self.data_privacy,        spd)
        self._d_dev   = lerp(self._d_dev,   self.device_integrity,    spd)
        self._d_aware = lerp(self._d_aware, self.awareness_level,     spd)
        self._d_rep   = lerp(self._d_rep,   self.digital_reputation,  spd)

    def apply(self, acct=0, priv=0, dev=0, aware=0, rep=0):
        self.account_security   = clamp(self.account_security   + acct)
        self.data_privacy       = clamp(self.data_privacy       + priv)
        self.device_integrity   = clamp(self.device_integrity   + dev)
        self.awareness_level    = clamp(self.awareness_level    + aware)
        self.digital_reputation = clamp(self.digital_reputation + rep)

    def average(self):
        return (self.account_security + self.data_privacy +
                self.device_integrity + self.awareness_level +
                self.digital_reputation) / 5.0

    @property
    def display(self):
        return [self._d_acct, self._d_priv, self._d_dev, self._d_aware, self._d_rep]

    def snap(self):
        self._d_acct  = self.account_security
        self._d_priv  = self.data_privacy
        self._d_dev   = self.device_integrity
        self._d_aware = self.awareness_level
        self._d_rep   = self.digital_reputation


# ─────────────────────────────────────────────────────────────
#  IMAGE MANAGER
# ─────────────────────────────────────────────────────────────

class ImageManager:
    _cache: Dict[str, np.ndarray] = {}
    IW, IH = 504, 190

    @classmethod
    def get(cls, key: str) -> np.ndarray:
        if key in cls._cache:
            return cls._cache[key]
        for ext in ['.png', '.jpg']:
            p = f"assets/{key}{ext}"
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    cls._cache[key] = cv2.resize(img, (cls.IW, cls.IH))
                    return cls._cache[key]
        img = cls._placeholder(key)
        cls._cache[key] = img
        return img

    @classmethod
    def _placeholder(cls, key: str) -> np.ndarray:
        img = np.zeros((cls.IH, cls.IW, 3), np.uint8)
        for row in range(cls.IH):
            t = row / cls.IH
            r = int(lerp(14, 35, t))
            g = int(lerp(18, 50, t))
            b = int(lerp(28, 72, t))
            img[row, :] = [b, g, r]
        for gx in range(0, cls.IW, 48):
            cv2.line(img, (gx, 0), (gx, cls.IH), (35, 55, 85), 1)
        for gy in range(0, cls.IH, 24):
            cv2.line(img, (0, gy), (cls.IW, gy), (35, 55, 85), 1)

        cx, cy = cls.IW // 2, cls.IH // 2

        if 'phishing' in key:
            cv2.rectangle(img, (cx-70, cy-28), (cx+70, cy+28), (60, 130, 200), 2)
            cv2.line(img, (cx-70, cy-28), (cx, cy+8), (60, 130, 200), 2)
            cv2.line(img, (cx+70, cy-28), (cx, cy+8), (60, 130, 200), 2)
            cv2.line(img, (cx+50, cy-50), (cx+72, cy-28), (60, 60, 220), 2)
            cv2.line(img, (cx+72, cy-50), (cx+50, cy-28), (60, 60, 220), 2)
            label = "PHISHING EMAIL"
        elif 'social' in key:
            for ox in [-52, 0, 52]:
                cv2.circle(img, (cx+ox, cy-18), 14, (100, 190, 255), 2)
                cv2.ellipse(img, (cx+ox, cy+20), (18, 12), 0, 180, 360, (100, 190, 255), 2)
            label = "SOCIAL MEDIA RISK"
        elif 'wifi' in key:
            for r in [16, 34, 52]:
                cv2.ellipse(img, (cx, cy+30), (r, r), 0, 205, 335, (100, 190, 255), 2)
            cv2.circle(img, (cx, cy+30), 5, (100, 190, 255), -1)
            cv2.putText(img, "?", (cx+34, cy-12), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 170, 60), 3, cv2.LINE_AA)
            label = "PUBLIC WIFI THREAT"
        elif 'ransomware' in key:
            cv2.rectangle(img, (cx-30, cy-8), (cx+30, cy+32), (40, 40, 180), -1)
            cv2.rectangle(img, (cx-30, cy-8), (cx+30, cy+32), (255, 170, 60), 2)
            cv2.ellipse(img, (cx, cy-18), (20, 20), 0, 180, 360, (255, 170, 60), 2)
            cv2.putText(img, "$", (cx-9, cy+24), cv2.FONT_HERSHEY_DUPLEX, 0.8, (235, 242, 255), 2, cv2.LINE_AA)
            label = "RANSOMWARE"
        elif 'identity' in key:
            cv2.rectangle(img, (cx-60, cy-32), (cx+60, cy+32), (25, 32, 48), -1)
            cv2.rectangle(img, (cx-60, cy-32), (cx+60, cy+32), (255, 170, 60), 2)
            cv2.circle(img, (cx-32, cy), 16, (100, 190, 255), 2)
            for lx in range(cx+2, cx+52, 14):
                cv2.line(img, (lx, cy-12), (lx+10, cy-12), (160, 180, 215), 1)
                cv2.line(img, (lx, cy-2), (lx+10, cy-2), (100, 120, 155), 1)
            label = "IDENTITY THEFT"
        elif 'breach' in key:
            pts = np.array([[cx, cy-42],[cx+36, cy-22],[cx+36, cy+16],
                            [cx, cy+42],[cx-36, cy+16],[cx-36, cy-22]], np.int32)
            cv2.polylines(img, [pts], True, (60, 60, 220), 2)
            cv2.line(img, (cx, cy-30), (cx-12, cy+4), (255, 170, 60), 3)
            cv2.line(img, (cx-12, cy+4), (cx+8, cy+4), (255, 170, 60), 3)
            cv2.line(img, (cx+8, cy+4), (cx, cy+30), (255, 170, 60), 3)
            label = "DATA BREACH"
        elif 'hygiene' in key:
            pts = np.array([[cx, cy-42],[cx+36, cy-22],[cx+36, cy+16],
                            [cx, cy+42],[cx-36, cy+16],[cx-36, cy-22]], np.int32)
            cv2.fillPoly(img, [pts], (20, 38, 22))
            cv2.polylines(img, [pts], True, (100, 210, 85), 2)
            cv2.line(img, (cx-16, cy+5), (cx, cy+20), (100, 210, 85), 3)
            cv2.line(img, (cx, cy+20), (cx+22, cy-10), (100, 210, 85), 3)
            label = "DIGITAL HYGIENE"
        else:
            label = key.upper()

        tw, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(img, label, (cls.IW//2 - tw[0]//2, cls.IH-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 120, 155), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (cls.IW-1, cls.IH-1), bgr(COLORS['accent_primary']), 1)
        return img


# ─────────────────────────────────────────────────────────────
#  SCENARIO DATA
# ─────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "phase_name": "PHASE 1  -  PHISHING ATTACK",
        "image_key":  "phishing_email",
        "context": (
            "You receive an urgent email from 'support@paypa1-secure.com' claiming "
            "your account will be locked in 24 hours. The email asks you to click a "
            "link and re-enter your credentials. The sender name shows 'PayPal Support' "
            "but the domain looks unusual. What do you do?"
        ),
        "decisions": [
            {
                "title":   "Click Link Immediately",
                "detail":  "Follow the email link and log in to fix the problem.",
                "impacts": {"acct": -30, "aware": -20, "rep": -10},
            },
            {
                "title":   "Inspect and Report",
                "detail":  "Check sender domain, report phishing, delete the email.",
                "impacts": {"acct": 25, "aware": 30, "rep": 10},
            },
            {
                "title":   "Ignore and Do Nothing",
                "detail":  "Don't click but also don't report. Just delete it.",
                "impacts": {"acct": -5, "aware": 5},
            },
        ],
        "fact": (
            "97% of people cannot reliably identify phishing emails. Attackers use "
            "domain spoofing (paypa1 vs paypal) to deceive victims. Always verify the "
            "sender's full domain before clicking any link. A 2023 Verizon report found "
            "36% of all data breaches involved phishing as the initial attack vector."
        ),
    },
    {
        "phase_name": "PHASE 2  -  SOCIAL MEDIA PRIVACY RISK",
        "image_key":  "social_media_risk",
        "context": (
            "Your social media profile is fully public. It shows your full name, city, "
            "employer, and daily routines. A friend posted that you are on vacation for "
            "2 weeks. Strangers can see your phone number in the About section. "
            "What action do you take?"
        ),
        "decisions": [
            {
                "title":   "Privatize and Audit",
                "detail":  "Set to private, review app permissions, remove personal data.",
                "impacts": {"priv": 30, "rep": 15, "aware": 15},
            },
            {
                "title":   "Stay Public for Networking",
                "detail":  "Keep public visibility. The benefits outweigh the risks.",
                "impacts": {"priv": -25, "rep": -10, "aware": -10},
            },
            {
                "title":   "Remove Suspicious Followers",
                "detail":  "Manually remove a few suspicious accounts but leave the rest.",
                "impacts": {"priv": 5, "aware": 5},
            },
        ],
        "fact": (
            "Over 15 million identity theft cases occur annually in the US alone. "
            "Oversharing location data and daily routines on social media is the #1 "
            "enabler of social engineering attacks. Conduct a privacy audit every 90 "
            "days and remove all contact information that is not strictly necessary."
        ),
    },
    {
        "phase_name": "PHASE 3  -  PUBLIC WIFI THREAT",
        "image_key":  "public_wifi",
        "context": (
            "You are at an international airport and need to check your bank balance "
            "and respond to work emails. The airport offers free WiFi: 'AirportFreeWifi'. "
            "You also see a second network called 'FREE_AIRPORT_WIFI' with a stronger "
            "signal. Your mobile data plan is running low. What do you do?"
        ),
        "decisions": [
            {
                "title":   "Connect With VPN Active",
                "detail":  "Use the official WiFi but tunnel all traffic through your VPN.",
                "impacts": {"dev": 25, "acct": 20, "aware": 20},
            },
            {
                "title":   "Connect Without Protection",
                "detail":  "Join the stronger signal network. Probably fine.",
                "impacts": {"dev": -30, "acct": -25, "priv": -20},
            },
            {
                "title":   "Use Mobile Hotspot",
                "detail":  "Create your own hotspot using mobile data only.",
                "impacts": {"dev": 20, "acct": 15, "priv": 10},
            },
        ],
        "fact": (
            "Man-in-the-Middle (MITM) attacks on public WiFi intercept data between you "
            "and the internet. 'Evil twin' rogue hotspots are common in airports. "
            "Attackers can capture credentials and session cookies in seconds. A trusted "
            "VPN encrypts your entire tunnel, making MITM attacks effectively useless. "
            "Never conduct financial transactions on unprotected public networks."
        ),
    },
    {
        "phase_name": "PHASE 5  -  LONG-TERM DIGITAL HYGIENE",
        "image_key":  "digital_hygiene",
        "context": (
            "After the security incident you realize your digital habits need a "
            "systematic overhaul. You are evaluating three long-term strategies to "
            "harden your digital life. Each requires time investment but offers "
            "different levels of protection. Which strategy do you commit to?"
        ),
        "decisions": [
            {
                "title":   "Enable MFA Everywhere",
                "detail":  "Activate multi-factor authentication on every account.",
                "impacts": {"acct": 35, "aware": 20, "rep": 15},
            },
            {
                "title":   "Deploy Password Manager",
                "detail":  "Set up a manager with unique 20-char passwords per site.",
                "impacts": {"acct": 20, "priv": 20, "aware": 15},
            },
            {
                "title":   "No Changes Needed",
                "detail":  "Things have been fine so far. No immediate action needed.",
                "impacts": {"acct": -20, "aware": -15, "rep": -15},
            },
        ],
        "fact": (
            "Microsoft research confirms MFA blocks 99.9% of automated account-takeover "
            "attacks. Yet only 57% of users have enabled it on their primary email. "
            "Password reuse affects 65% of people. A single breach cascades across "
            "dozens of accounts. Both MFA and a password manager together eliminate over "
            "95% of common attack vectors targeting individuals."
        ),
    },
]

CRISIS_EVENTS = [
    {
        "name":      "RANSOMWARE INFECTION",
        "image_key": "ransomware_attack",
        "description": (
            "ALERT: A ransomware payload has been detected on your device. Files are "
            "encrypted and held for $2,400 in cryptocurrency. System access is "
            "restricted. Backup integrity is unknown."
        ),
        "base_impacts":    {"acct": -10, "priv": -25, "dev": -40, "rep": -20},
        "mitigation_stat": "device_integrity",
        "fact": (
            "Global ransomware damage exceeded $30 billion in 2023. Paying the ransom "
            "does NOT guarantee file recovery. 42% of paying victims never get their "
            "data back. The best defenses are regular offline backups (3-2-1 rule), "
            "keeping your OS patched, and never opening unexpected email attachments. "
            "Ransomware exploits unpatched vulnerabilities 60% of the time."
        ),
    },
    {
        "name":      "IDENTITY THEFT ALERT",
        "image_key": "identity_theft",
        "description": (
            "ALERT: Your personal information was used to open 3 fraudulent credit "
            "accounts. Your SSN, DOB, and home address have been found in underground "
            "markets. Unauthorized charges are appearing on your bank accounts."
        ),
        "base_impacts":    {"acct": -30, "priv": -35, "aware": -10, "rep": -25},
        "mitigation_stat": "data_privacy",
        "fact": (
            "Identity theft victims spend an average of 200 hours recovering their "
            "identity. The dark web hosts over 15 billion stolen credentials. Credit "
            "freezes are FREE in the US and prevent unauthorized account openings. "
            "Monitor your credit report weekly and set up fraud alerts. Early detection "
            "reduces recovery time by 80%."
        ),
    },
    {
        "name":      "ACCOUNT DATA BREACH",
        "image_key": "data_breach",
        "description": (
            "ALERT: A major platform you use confirmed a breach exposing 182 million "
            "user records. Your email, hashed password, location history, and payment "
            "metadata are confirmed as leaked in this incident."
        ),
        "base_impacts":    {"acct": -30, "priv": -30, "aware": -15, "rep": -15},
        "mitigation_stat": "account_security",
        "fact": (
            "Have I Been Pwned tracks over 12 billion breached accounts. When a breach "
            "occurs, change your password immediately on that site AND every site where "
            "you reused it. Password reuse is how breaches cascade. Enable breach "
            "notification services. A leaked hashed password can be cracked in seconds "
            "if you used a common word. Use 16-character random passphrases."
        ),
    },
]

GRADE_DATA = [
    (80, "CYBER SECURITY CHAMPION",  (100, 210, 85),
     "Exceptional performance. You demonstrated sophisticated threat awareness and "
     "applied defense-in-depth principles under pressure. Your digital hygiene meets "
     "enterprise security standards. Consider a formal cybersecurity certification."),
    (65, "DIGITALLY RESPONSIBLE",    (75, 185, 105),
     "Solid security posture with room for improvement. You identified most major "
     "threats correctly but some decision gaps remain. Focus on enabling MFA across "
     "all accounts and auditing social media privacy quarterly."),
    (50, "AT RISK USER",             (75, 170, 215),
     "Significant security gaps identified. Several choices exposed you to preventable "
     "threats. Immediate action: enable MFA on email and banking, install a reputable "
     "password manager, set social media to private, and install a trusted VPN."),
    ( 0, "CRITICAL VULNERABILITY",   (60, 60, 215),
     "URGENT: Your digital habits create severe exposure to identity theft, financial "
     "fraud, and data loss. Change all passwords to unique 16+ character strings, "
     "enable MFA everywhere, run a full antivirus scan, and freeze your credit."),
]

def get_grade(score):
    for (min_s, name, color, analysis) in GRADE_DATA:
        if score >= min_s:
            return name, color, analysis
    return GRADE_DATA[-1][1], GRADE_DATA[-1][2], GRADE_DATA[-1][3]


# ─────────────────────────────────────────────────────────────
#  PHASE ENUM
# ─────────────────────────────────────────────────────────────

class Phase(Enum):
    SCENARIO    = auto()
    FACT        = auto()
    CRISIS      = auto()
    CRISIS_FACT = auto()
    FINAL       = auto()


# ─────────────────────────────────────────────────────────────
#  DECISION CARD
# ─────────────────────────────────────────────────────────────

@dataclass
class Card:
    idx:     int
    title:   str
    detail:  str
    impacts: dict
    x: int
    y: int
    w: int
    h: int
    hover_prog: float = 0.0
    hovered:    bool  = False

    IMPACT_KEY_MAP = {
        "acct":  "Account Security",
        "priv":  "Data Privacy",
        "dev":   "Device Integrity",
        "aware": "Awareness Level",
        "rep":   "Digital Reputation",
    }

    def impact_lines(self):
        lines = []
        for k, v in self.impacts.items():
            label = self.IMPACT_KEY_MAP.get(k, k)
            sign  = "+" if v >= 0 else ""
            lines.append((f"{sign}{v}  {label}", v >= 0))
        return lines


# ─────────────────────────────────────────────────────────────
#  MAIN SIMULATOR
# ─────────────────────────────────────────────────────────────

HOVER_FRAMES = int(2.0 * FPS)
FACT_FRAMES  = int(2.5 * FPS)

class Sim:

    def __init__(self):
        self.canvas = np.zeros((H, W, 3), np.uint8)
        self.stats  = Stats()
        self.mx     = 0
        self.my     = 0
        self.frame  = 0

        self.scenario_idx = 0
        self.phase        = Phase.SCENARIO
        self.cards: List[Card] = []

        self.fact_text  = ""
        self.fact_key   = ""
        self.fact_timer = 0

        self.crisis       = None
        self.crisis_timer = 0

        self.hovered_idx: Optional[int] = None

        self.restart_prog    = 0.0
        self.restart_hovered = False

        self._build_cards()

    # ── Card builder ───────────────────────────────────────────

    def _build_cards(self):
        if self.scenario_idx >= len(SCENARIOS):
            return
        decs  = SCENARIOS[self.scenario_idx]["decisions"]
        n     = len(decs)
        total = n * CARD_W + (n - 1) * CARD_GAP
        sx    = (W - total) // 2

        self.cards = []
        for i, d in enumerate(decs):
            cx = sx + i * (CARD_W + CARD_GAP)
            self.cards.append(Card(
                idx=i, title=d["title"], detail=d["detail"],
                impacts=d["impacts"],
                x=cx, y=DECK_Y + 6, w=CARD_W, h=CARD_H
            ))

    # ── Phase transitions ──────────────────────────────────────

    def _choose_card(self, card: Card):
        self.stats.apply(**card.impacts)
        sc = SCENARIOS[self.scenario_idx]
        self.fact_text  = sc["fact"]
        self.fact_key   = sc["image_key"]
        self.fact_timer = 0
        self.hovered_idx = None
        self.phase = Phase.FACT

    def _end_fact(self):
        self.scenario_idx += 1
        if self.scenario_idx == 3:
            self._start_crisis()
        elif self.scenario_idx >= len(SCENARIOS):
            self.phase = Phase.FINAL
        else:
            self._build_cards()
            self.phase = Phase.SCENARIO

    def _start_crisis(self):
        self.crisis       = random.choice(CRISIS_EVENTS)
        self.crisis_timer = 0
        self.phase        = Phase.CRISIS

        mit_stat = self.crisis["mitigation_stat"]
        stat_val = getattr(self.stats, mit_stat)
        factor   = 1.0 - (stat_val / 100.0) * 0.65
        impacts  = {}
        for k, v in self.crisis["base_impacts"].items():
            impacts[k] = int(v * factor) if v < 0 else v
        self.stats.apply(**impacts)

    def _end_crisis_alert(self):
        self.crisis_timer = 0
        self.phase = Phase.CRISIS_FACT

    def _end_crisis_fact(self):
        self._build_cards()
        self.phase = Phase.SCENARIO

    def _restart(self):
        self.__init__()

    # ── Mouse ──────────────────────────────────────────────────

    def on_mouse(self, event, x, y, flags, param):
        self.mx = x
        self.my = y

    # ── Update ─────────────────────────────────────────────────

    def update(self):
        self.stats.smooth()
        self.frame += 1

        if self.phase == Phase.SCENARIO:
            self._update_scenario()
        elif self.phase == Phase.FACT:
            self.fact_timer += 1
            if self.fact_timer >= FACT_FRAMES:
                self._end_fact()
        elif self.phase == Phase.CRISIS:
            self.crisis_timer += 1
            if self.crisis_timer >= int(2.0 * FPS):
                self._end_crisis_alert()
        elif self.phase == Phase.CRISIS_FACT:
            self.crisis_timer += 1
            if self.crisis_timer >= FACT_FRAMES:
                self._end_crisis_fact()

    def _update_scenario(self):
        chosen      = None
        new_hovered = None

        for card in self.cards:
            if point_in_rect(self.mx, self.my, card.x, card.y, card.w, card.h):
                card.hovered    = True
                card.hover_prog = min(1.0, card.hover_prog + 1 / HOVER_FRAMES)
                new_hovered     = card.idx
                if card.hover_prog >= 1.0:
                    chosen = card
                    break
            else:
                card.hovered    = False
                card.hover_prog = max(0.0, card.hover_prog - 3 / HOVER_FRAMES)

        self.hovered_idx = new_hovered
        if chosen:
            self._choose_card(chosen)

    # ── Render ─────────────────────────────────────────────────

    def render(self):
        self.canvas[:] = bgr(COLORS['bg_primary'])

        if self.phase in (Phase.SCENARIO, Phase.FACT):
            self._draw_header()
            self._draw_left()
            self._draw_center()
            self._draw_right()
            if self.phase == Phase.SCENARIO:
                self._draw_deck()
            else:
                self._draw_fact_screen()

        elif self.phase == Phase.CRISIS:
            self._draw_header()
            self._draw_left()
            self._draw_crisis_panel()

        elif self.phase == Phase.CRISIS_FACT:
            self._draw_header()
            self._draw_left()
            self._draw_crisis_fact()

        elif self.phase == Phase.FINAL:
            self._draw_final()

    # ── HEADER ─────────────────────────────────────────────────

    def _draw_header(self):
        panel(self.canvas, 0, 0, W, HEADER_H, COLORS['bg_secondary'], alpha=0.98)

        put_bold(self.canvas, "CYBERSEC AWARENESS SIMULATOR",
                 (18, 38), COLORS['accent_primary'], 0.90, 2)

        if self.scenario_idx < len(SCENARIOS):
            phase_label = SCENARIOS[self.scenario_idx]["phase_name"]
        elif self.phase in (Phase.CRISIS, Phase.CRISIS_FACT):
            phase_label = "PHASE 4  -  CYBER CRISIS EVENT"
        else:
            phase_label = "FINAL EVALUATION"

        put(self.canvas, phase_label, (20, 60), COLORS['accent_secondary'], 0.52, 1)

        # Frame counter
        put(self.canvas, f"FRAME {self.frame:06d}",
            (W - 175, 22), COLORS['text_dim'], 0.38, 1)

        # Phase progress dots
        for i in range(5):
            dx   = W - 170 + i * 30
            done = (self.scenario_idx > i)
            col  = COLORS['accent_primary'] if done else COLORS['text_dim']
            cv2.circle(self.canvas, (dx, 52), 6, bgr(col), -1 if done else 1)

        hline(self.canvas, 0, W, HEADER_H - 1, COLORS['accent_primary'], 2)

    # ── LEFT PANEL ─────────────────────────────────────────────

    def _draw_left(self):
        x, y, w, h = LEFT_X, COL_Y, LEFT_W, COL_H
        panel(self.canvas, x, y, w, h, COLORS['bg_panel'],
              border=COLORS['accent_primary'], alpha=0.97)

        put_bold(self.canvas, "SECURITY METRICS",
                 (x+12, y+22), COLORS['accent_primary'], 0.50, 1)
        hline(self.canvas, x+8, x+w-8, y+30, COLORS['accent_primary'])

        bw  = w - 46
        bh  = 13
        bx  = x + 14
        gap = (h - 52) // 5

        for i, (label, val) in enumerate(zip(STAT_LABELS, self.stats.display)):
            by = y + 46 + i * gap
            put(self.canvas, label, (bx, by - 6), COLORS['text_secondary'], 0.42, 1)
            stat_bar(self.canvas, bx, by, bw, bh, val)
            put(self.canvas, f"{int(val)}",
                (bx + bw + 5, by + bh), COLORS['text_secondary'], 0.42, 1)

        score = self.stats.average()
        put(self.canvas, f"SCORE: {score:.1f}",
            (x+14, y+h-14), COLORS['text_dim'], 0.43, 1)

        # Animated border
        p  = 0.55 + 0.45 * math.sin(self.frame * 0.05)
        bc = tuple(int(COLORS['bg_panel'][j] + (COLORS['accent_primary'][j] - COLORS['bg_panel'][j]) * p)
                   for j in range(3))
        border_rect(self.canvas, x, y, w, h, bc)

    # ── CENTER PANEL ───────────────────────────────────────────

    def _draw_center(self):
        x, y, w, h = MID_X, COL_Y, MID_W, COL_H
        panel(self.canvas, x, y, w, h, COLORS['bg_panel'],
              border=COLORS['accent_secondary'], alpha=0.97)

        put_bold(self.canvas, "SCENARIO BRIEFING",
                 (x+14, y+22), COLORS['accent_secondary'], 0.52, 1)
        hline(self.canvas, x+8, x+w-8, y+30, COLORS['accent_secondary'])

        if self.scenario_idx >= len(SCENARIOS):
            return

        sc = SCENARIOS[self.scenario_idx]

        # Image — fixed height, no overflow
        margin = 8
        img_y  = y + 38
        img_w  = w - margin * 2
        img_h  = ImageManager.IH
        img    = ImageManager.get(sc["image_key"])
        thumb  = cv2.resize(img, (img_w, img_h))

        # Clamp to canvas
        y1 = img_y
        y2 = min(y1 + img_h, H)
        x1 = x + margin
        x2 = min(x1 + img_w, W)
        if y2 > y1 and x2 > x1:
            self.canvas[y1:y2, x1:x2] = thumb[:y2-y1, :x2-x1]
            border_rect(self.canvas, x1, y1, x2-x1, y2-y1, COLORS['accent_secondary'])

        # Context text
        text_y = img_y + img_h + 18
        lines  = wrap_text(sc["context"], 60)
        for i, line in enumerate(lines[:6]):
            ty = text_y + i * 22
            if ty < y + h - 28:
                put(self.canvas, line, (x+14, ty), COLORS['text_primary'], 0.46, 1)

        put(self.canvas, "HOVER A DECISION CARD FOR 2 SEC TO CONFIRM",
            (x+14, y+h-12), COLORS['text_dim'], 0.37, 1)

    # ── RIGHT PANEL ────────────────────────────────────────────

    def _draw_right(self):
        x, y, w, h = RIGHT_X, COL_Y, RIGHT_W, COL_H
        panel(self.canvas, x, y, w, h, COLORS['bg_panel'],
              border=COLORS['text_dim'], alpha=0.97)

        put_bold(self.canvas, "IMPACT PREVIEW",
                 (x+12, y+22), COLORS['text_secondary'], 0.50, 1)
        hline(self.canvas, x+8, x+w-8, y+30, COLORS['text_dim'])

        if self.hovered_idx is None or not self.cards:
            put(self.canvas, "Hover a decision card",
                (x+12, y+55), COLORS['text_dim'], 0.43, 1)
            put(self.canvas, "to preview its impact",
                (x+12, y+73), COLORS['text_dim'], 0.43, 1)
            put(self.canvas, "on your security stats.",
                (x+12, y+91), COLORS['text_dim'], 0.43, 1)
            return

        card = self.cards[self.hovered_idx]

        # Title
        tlines = wrap_text(card.title, 26)
        for i, l in enumerate(tlines):
            put_bold(self.canvas, l, (x+12, y+50+i*22),
                     COLORS['accent_primary'], 0.52, 1)

        # Detail
        dlines = wrap_text(card.detail, 28)
        for i, l in enumerate(dlines[:3]):
            put(self.canvas, l, (x+12, y+96+i*18), COLORS['text_secondary'], 0.40, 1)

        hline(self.canvas, x+8, x+w-8, y+155, COLORS['text_dim'])
        put(self.canvas, "STAT CHANGES:", (x+12, y+170), COLORS['text_dim'], 0.40, 1)

        for i, (txt, pos) in enumerate(card.impact_lines()):
            col = COLORS['stat_excellent'] if pos else COLORS['stat_critical']
            put(self.canvas, txt, (x+14, y+188+i*20), col, 0.42, 1)

        # Hold bar
        by = y + h - 42
        put(self.canvas, "HOLD TO CONFIRM:", (x+12, by), COLORS['text_dim'], 0.37, 1)
        progress_bar(self.canvas, x+10, by+10, w-20, 10,
                     card.hover_prog, COLORS['accent_primary'])
        put(self.canvas, f"{int(card.hover_prog*100)}%",
            (x+w-34, by+20), COLORS['text_dim'], 0.38, 1)

    # ── DECISION DECK ──────────────────────────────────────────

    def _draw_deck(self):
        panel(self.canvas, 0, DECK_Y, W, DECK_H,
              COLORS['bg_secondary'], alpha=0.95)
        hline(self.canvas, 0, W, DECK_Y, COLORS['accent_primary'], 2)

        put(self.canvas, "SELECT YOUR RESPONSE  -  HOLD CARD FOR 2 SECONDS",
            (16, DECK_Y + 16), COLORS['accent_primary'], 0.45, 1)

        for card in self.cards:
            self._draw_card(card)

    def _draw_card(self, card: Card):
        x, y, w, h = card.x, card.y, card.w, card.h

        # Shadow
        fill_rect(self.canvas, x+3, y+3, w, h, (8, 10, 16), 0.6)

        # Base
        panel(self.canvas, x, y, w, h, COLORS['bg_panel'],
              border=COLORS['accent_secondary'] if card.hovered else COLORS['accent_primary'],
              bthick=2 if card.hovered else 1, alpha=0.97)

        # Hover tint
        if card.hovered and card.hover_prog > 0:
            fill_rect(self.canvas, x+1, y+1, w-2, h-2,
                      COLORS['hover_overlay'], card.hover_prog * 0.28)

        # Number badge
        cv2.circle(self.canvas, (x+24, y+24), 13, bgr(COLORS['accent_primary']), -1)
        put_bold(self.canvas, str(card.idx+1), (x+18, y+30),
                 COLORS['bg_primary'], 0.50, 2)

        # Title
        tlines = wrap_text(card.title, 36)
        for i, l in enumerate(tlines):
            put_bold(self.canvas, l, (x+44, y+22+i*21),
                     COLORS['text_primary'], 0.54, 1)

        hline(self.canvas, x+8, x+w-8, y+54, COLORS['text_dim'])

        # Detail
        dlines = wrap_text(card.detail, 46)
        for i, l in enumerate(dlines[:2]):
            put(self.canvas, l, (x+10, y+68+i*18), COLORS['text_secondary'], 0.40, 1)

        hline(self.canvas, x+8, x+w-8, y+110, COLORS['text_dim'])

        # Impact lines — 2 columns
        impacts = card.impact_lines()
        for i, (txt, pos) in enumerate(impacts[:4]):
            col   = COLORS['stat_excellent'] if pos else COLORS['stat_critical']
            col_x = x + 10 + (i % 2) * 190
            col_y = y + 126 + (i // 2) * 20
            if col_y < y + h - 25:
                put(self.canvas, txt, (col_x, col_y), col, 0.40, 1)

        # Hold bar at bottom of card
        bx, by = x+10, y+h-22
        bw     = w - 20
        progress_bar(self.canvas, bx, by, bw, 12, card.hover_prog, COLORS['accent_primary'])
        if card.hovered and self.frame % 40 < 28:
            pct = f"{int(card.hover_prog*100)}%"
            put(self.canvas, f"HOLD  {pct}", (bx+4, by+9),
                COLORS['bg_primary'], 0.36, 1)

    # ── FACT SCREEN ────────────────────────────────────────────

    def _draw_fact_screen(self):
        fill_rect(self.canvas, 0, 0, W, H, COLORS['bg_primary'], 0.82)

        fx, fy, fw, fh = 100, 75, 1080, 440
        panel(self.canvas, fx, fy, fw, fh, COLORS['bg_panel'],
              border=COLORS['accent_secondary'], bthick=2, alpha=0.98)

        put_bold(self.canvas, "EDUCATIONAL INSIGHT",
                 (fx+20, fy+34), COLORS['accent_secondary'], 0.72, 2)
        hline(self.canvas, fx+12, fx+fw-12, fy+46, COLORS['accent_secondary'], 2)

        # Thumbnail
        img   = ImageManager.get(self.fact_key)
        thumb = cv2.resize(img, (230, 130))
        tx, ty = fx+fw-248, fy+60
        self.canvas[ty:ty+130, tx:tx+230] = thumb
        border_rect(self.canvas, tx, ty, 230, 130, COLORS['accent_secondary'])

        # Fact text
        lines = wrap_text(self.fact_text, 70)
        for i, line in enumerate(lines[:10]):
            ty2 = fy + 64 + i * 28
            if ty2 < fy + fh - 42:
                put(self.canvas, line, (fx+20, ty2), COLORS['text_primary'], 0.48, 1)

        # Progress bar
        prog = min(1.0, self.fact_timer / FACT_FRAMES)
        bx, by = fx+16, fy+fh-32
        bw     = fw - 32
        progress_bar(self.canvas, bx, by, bw, 14, prog, COLORS['accent_secondary'])
        put(self.canvas, f"AUTO-ADVANCING  {int(prog*100)}%",
            (bx+6, by+10), COLORS['bg_primary'], 0.38, 1)

    # ── CRISIS PANEL ───────────────────────────────────────────

    def _draw_crisis_panel(self):
        if not self.crisis:
            return
        pulse = int(100 + 100 * math.sin(self.frame * 0.18))
        cv2.rectangle(self.canvas, (1, HEADER_H+1), (W-2, H-2), (0, 0, pulse), 3)

        cx, cy, cw, ch = 120, COL_Y, 1040, 430
        panel(self.canvas, cx, cy, cw, ch, COLORS['bg_panel'],
              border=COLORS['accent_alert'], bthick=3, alpha=0.98)

        put_bold(self.canvas, f"!! CYBER CRISIS: {self.crisis['name']}",
                 (cx+16, cy+34), COLORS['accent_alert'], 0.75, 2)
        hline(self.canvas, cx+12, cx+cw-12, cy+46, COLORS['accent_alert'], 2)

        img   = ImageManager.get(self.crisis['image_key'])
        thumb = cv2.resize(img, (330, 168))
        ix, iy = cx+cw-348, cy+58
        self.canvas[iy:iy+168, ix:ix+330] = thumb
        border_rect(self.canvas, ix, iy, 330, 168, COLORS['accent_alert'], 2)

        lines = wrap_text(self.crisis['description'], 62)
        for i, l in enumerate(lines[:5]):
            put(self.canvas, l, (cx+16, cy+62+i*26), COLORS['text_primary'], 0.47, 1)

        put(self.canvas, "SECURITY METRICS IMPACTED  -  MITIGATION APPLIED",
            (cx+16, cy+200), COLORS['stat_critical'], 0.42, 1)

        key_labels = {"acct": STAT_LABELS[0], "priv": STAT_LABELS[1],
                      "dev": STAT_LABELS[2], "aware": STAT_LABELS[3], "rep": STAT_LABELS[4]}
        for i, (k, v) in enumerate(self.crisis['base_impacts'].items()):
            label = key_labels.get(k, k)
            sign  = "+" if v >= 0 else ""
            col   = COLORS['stat_excellent'] if v >= 0 else COLORS['stat_critical']
            col_x = cx + 16 + (i % 2) * 340
            col_y = cy + 222 + (i // 2) * 22
            put(self.canvas, f"{sign}{v}  {label}", (col_x, col_y), col, 0.43, 1)

        mit = self.crisis['mitigation_stat'].replace("_", " ").title()
        put(self.canvas, f"Mitigation: High {mit} reduced damage severity",
            (cx+16, cy+ch-50), COLORS['text_secondary'], 0.40, 1)

        prog = min(1.0, self.crisis_timer / int(2.0*FPS))
        progress_bar(self.canvas, cx+16, cy+ch-30, cw-32, 12, prog, COLORS['accent_alert'])
        put(self.canvas, f"ASSESSING DAMAGE  {int(prog*100)}%",
            (cx+22, cy+ch-22), COLORS['bg_primary'], 0.38, 1)

    # ── CRISIS FACT ────────────────────────────────────────────

    def _draw_crisis_fact(self):
        if not self.crisis:
            return
        fill_rect(self.canvas, 0, HEADER_H, W, H-HEADER_H, COLORS['bg_primary'], 0.88)

        fx, fy, fw, fh = 80, COL_Y, 1120, 440
        panel(self.canvas, fx, fy, fw, fh, COLORS['bg_panel'],
              border=COLORS['accent_alert'], bthick=2, alpha=0.98)

        put_bold(self.canvas, f"CRISIS FACT  -  {self.crisis['name']}",
                 (fx+18, fy+34), COLORS['accent_alert'], 0.68, 2)
        hline(self.canvas, fx+12, fx+fw-12, fy+46, COLORS['accent_alert'], 2)

        lines = wrap_text(self.crisis['fact'], 82)
        for i, l in enumerate(lines[:10]):
            ty = fy + 65 + i * 30
            if ty < fy + fh - 42:
                put(self.canvas, l, (fx+18, ty), COLORS['text_primary'], 0.48, 1)

        prog = min(1.0, self.crisis_timer / FACT_FRAMES)
        bx, by = fx+16, fy+fh-32
        bw     = fw - 32
        progress_bar(self.canvas, bx, by, bw, 14, prog, COLORS['accent_alert'])
        put(self.canvas, f"CONTINUING TO PHASE 5  {int(prog*100)}%",
            (bx+6, by+10), COLORS['text_primary'], 0.38, 1)

    # ── FINAL SCREEN ───────────────────────────────────────────

    def _draw_final(self):
        self.canvas[:] = bgr(COLORS['bg_primary'])

        score              = self.stats.average()
        name, color, analysis = get_grade(score)

        # Banner
        fill_rect(self.canvas, 0, 0, W, 90, color, 1.0)
        put_bold(self.canvas, "CYBERSEC SIMULATION  -  FINAL EVALUATION",
                 (20, 36), COLORS['bg_primary'], 0.80, 2)
        put_bold(self.canvas, name, (20, 72), COLORS['bg_primary'], 0.68, 2)

        put_bold(self.canvas, f"OVERALL SCORE:  {score:.1f} / 100",
                 (20, 116), COLORS['accent_primary'], 0.80, 2)

        al = wrap_text(analysis, 80)
        for i, l in enumerate(al[:5]):
            put(self.canvas, l, (20, 146+i*24), COLORS['text_primary'], 0.47, 1)

        hline(self.canvas, 10, W-10, 278, COLORS['accent_primary'])

        put_bold(self.canvas, "FINAL SECURITY PROFILE",
                 (20, 304), COLORS['accent_secondary'], 0.54, 1)

        for i, (label, val) in enumerate(zip(STAT_LABELS, self.stats.display)):
            bx, by = 20, 320 + i * 36
            put(self.canvas, label, (bx, by), COLORS['text_secondary'], 0.44, 1)
            stat_bar(self.canvas, bx, by+6, 500, 16, val)
            put_bold(self.canvas, str(int(val)), (bx+510, by+18),
                     COLORS['text_primary'], 0.48, 1)

        self._draw_radar(890, 450, 150)

        # Restart button
        self._update_restart()
        rx, ry, rw, rh = W//2-130, H-74, 260, 50
        fill_rect(self.canvas, rx, ry, rw, rh, COLORS['bg_secondary'], 0.98)
        if self.restart_prog > 0:
            fc = COLORS['stat_excellent'] if self.restart_prog < 1.0 else color
            fill_rect(self.canvas, rx, ry, int(rw*self.restart_prog), rh, fc, 1.0)
        border_rect(self.canvas, rx, ry, rw, rh, COLORS['accent_primary'], 2)
        lbl = "HOLD TO RESTART" if self.restart_prog < 1.0 else "RESTARTING..."
        put_bold(self.canvas, lbl, (rx+rw//2-66, ry+32),
                 COLORS['text_primary'], 0.54, 1)

    def _update_restart(self):
        rx, ry, rw, rh = W//2-130, H-74, 260, 50
        if point_in_rect(self.mx, self.my, rx, ry, rw, rh):
            self.restart_prog    = min(1.0, self.restart_prog + 1/FPS)
            self.restart_hovered = True
            if self.restart_prog >= 1.0:
                self._restart()
        else:
            self.restart_prog    = max(0.0, self.restart_prog - 2/FPS)
            self.restart_hovered = False

    def _draw_radar(self, cx, cy, radius):
        n    = 5
        disp = self.stats.display

        for pct in [0.25, 0.50, 0.75, 1.0]:
            pts = []
            for i in range(n):
                a = math.pi/2 + 2*math.pi*i/n
                r = radius * pct
                pts.append((int(cx + r*math.cos(a)), int(cy - r*math.sin(a))))
            cv2.polylines(self.canvas, [np.array(pts, np.int32)], True,
                          bgr(COLORS['text_dim']), 1)

        for i in range(n):
            a = math.pi/2 + 2*math.pi*i/n
            cv2.line(self.canvas, (cx, cy),
                     (int(cx + radius*math.cos(a)), int(cy - radius*math.sin(a))),
                     bgr(COLORS['text_dim']), 1)

        data_pts = []
        for i, v in enumerate(disp):
            a = math.pi/2 + 2*math.pi*i/n
            r = radius * v / 100.0
            data_pts.append((int(cx + r*math.cos(a)), int(cy - r*math.sin(a))))
        dp = np.array(data_pts, np.int32)

        overlay = self.canvas.copy()
        cv2.fillPoly(overlay, [dp], bgr(COLORS['accent_secondary']))
        self.canvas[:] = cv2.addWeighted(self.canvas, 0.72, overlay, 0.28, 0)
        cv2.polylines(self.canvas, [dp], True, bgr(COLORS['accent_secondary']), 2)
        for pt in data_pts:
            cv2.circle(self.canvas, pt, 5, bgr(COLORS['accent_primary']), -1)

        for i, lbl in enumerate(STAT_LABELS):
            a  = math.pi/2 + 2*math.pi*i/n
            lx = int(cx + (radius+28)*math.cos(a))
            ly = int(cy - (radius+28)*math.sin(a))
            words = lbl.split()
            for j, w in enumerate(words):
                tw, _ = cv2.getTextSize(w, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                put(self.canvas, w, (lx - tw[0]//2, ly+j*13),
                    COLORS['text_secondary'], 0.36, 1)

    # ── Main loop ──────────────────────────────────────────────

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, W, H)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        delay = max(1, 1000 // FPS)
        print("=" * 56)
        print("  CYBERSEC AWARENESS SIMULATOR  v2.0")
        print("  1280x720  |  Hover cards 2 sec to confirm")
        print("  ESC or Q to quit")
        print("=" * 56)

        while True:
            self.update()
            self.render()
            cv2.imshow(WINDOW, self.canvas)
            if cv2.waitKey(delay) & 0xFF in (27, ord('q'), ord('Q')):
                break

        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Sim().run()