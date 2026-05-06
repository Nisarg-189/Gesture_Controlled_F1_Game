"""
GripAI — HUD
Heads-up display: speed, lap times, AI adaptation meter, race states.
"""

import math
import time
import numpy as np
import cv2
from typing import List, Optional


def _fmt_time(t: float) -> str:
    if t == float("inf"):
        return "--:--.---"
    m = int(t // 60)
    s = t % 60
    return f"{m}:{s:06.3f}"


class HUD:
    """Renders all overlays onto the game canvas."""

    ACCENT = (0, 200, 120)        # GripAI green
    WARNING = (0, 80, 220)        # warm orange-ish (BGR)
    WHITE = (240, 240, 240)
    DARK = (15, 15, 20)
    AI_COLOR = (255, 180, 50)

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def draw_menu(self, canvas: np.ndarray):
        self._panel(canvas, self.w // 2 - 260, self.h // 2 - 180, 520, 360)
        self._text(canvas, "GRIPAI", self.w // 2, self.h // 2 - 120,
                   scale=2.8, color=self.ACCENT, center=True, thickness=3)
        self._text(canvas, "Gesture-Controlled F1 Racing", self.w // 2, self.h // 2 - 60,
                   scale=0.72, center=True)
        self._text(canvas, "PRESS  SPACE / ENTER  TO RACE", self.w // 2, self.h // 2 + 10,
                   scale=0.62, color=self.WHITE, center=True)
        self._divider(canvas, self.h // 2 + 45)
        controls = [
            ("✊ Fist", "Brake"),
            ("✋ Open", "Accelerate"),
            ("👈 Tilt", "Steer"),
            ("WASD", "Keyboard fallback"),
        ]
        for i, (gesture, action) in enumerate(controls):
            x = self.w // 2 - 200 + (i % 2) * 200
            y = self.h // 2 + 85 + (i // 2) * 34
            self._text(canvas, f"{gesture}  →  {action}", x, y, scale=0.52, center=False)

    # ------------------------------------------------------------------
    # Countdown
    # ------------------------------------------------------------------

    def draw_countdown(self, canvas: np.ndarray, remaining: float):
        value = math.ceil(remaining)
        label = str(value) if value > 0 else "GO!"
        color = self.ACCENT if value == 0 else self.WHITE
        scale = 5.0 + (1.0 - (remaining % 1.0)) * 1.5
        self._text(canvas, label, self.w // 2, self.h // 2 + 30,
                   scale=scale, color=color, center=True, thickness=5)

    # ------------------------------------------------------------------
    # Racing
    # ------------------------------------------------------------------

    def draw_racing(
        self,
        canvas: np.ndarray,
        player_speed: float,
        race_time: float,
        lap_time: float,
        player_lap: int,
        ai_lap: int,
        total_laps: int,
        player_best: float,
        ai_adaptation: float,
    ):
        # Speed (top-left)
        kmh = player_speed * 3.6
        self._panel(canvas, 10, 10, 180, 80)
        self._text(canvas, f"{kmh:.0f}", 100, 55, scale=2.0, color=self.ACCENT,
                   center=True, thickness=2)
        self._text(canvas, "km/h", 100, 78, scale=0.5, center=True)

        # Lap counter (top-centre)
        self._panel(canvas, self.w // 2 - 120, 10, 240, 55)
        self._text(canvas, f"LAP  {player_lap} / {total_laps}",
                   self.w // 2, 47, scale=0.8, color=self.WHITE, center=True)

        # Timer (top-right)
        self._panel(canvas, self.w - 230, 10, 220, 55)
        self._text(canvas, _fmt_time(race_time), self.w - 120, 47,
                   scale=0.72, color=self.ACCENT, center=True)

        # Lap time (below timer)
        self._panel(canvas, self.w - 230, 70, 220, 45)
        self._text(canvas, f"LAP  {_fmt_time(lap_time)}", self.w - 120, 103,
                   scale=0.58, color=self.WHITE, center=True)

        # Best lap
        self._panel(canvas, self.w - 230, 120, 220, 40)
        best_str = _fmt_time(player_best) if player_best < float("inf") else "--"
        self._text(canvas, f"BEST  {best_str}", self.w - 120, 149,
                   scale=0.55, color=(80, 220, 80), center=True)

        # AI adaptation meter (bottom-left)
        self._draw_adaptation_bar(canvas, ai_adaptation)

        # Throttle / brake bars
        # (placeholder — can read car.throttle/brake directly if passed)

    # ------------------------------------------------------------------
    # AI Adaptation bar
    # ------------------------------------------------------------------

    def _draw_adaptation_bar(self, canvas: np.ndarray, level: float):
        x0, y0 = 10, self.h - 80
        self._panel(canvas, x0, y0, 220, 65)
        self._text(canvas, "AI ADAPTATION", x0 + 110, y0 + 18,
                   scale=0.45, color=self.AI_COLOR, center=True)
        # Bar background
        bx, by, bw, bh = x0 + 10, y0 + 28, 200, 14
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
        fill = int(bw * min(1.0, level))
        if fill > 0:
            # Colour from green → yellow → red as adaptation grows
            r = int(255 * min(1.0, level * 2))
            g = int(255 * min(1.0, (1.0 - level) * 2))
            cv2.rectangle(canvas, (bx, by), (bx + fill, by + bh), (0, g, r), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (80, 80, 80), 1)
        pct = int(level * 100)
        self._text(canvas, f"{pct}%  learned", x0 + 110, y0 + 55,
                   scale=0.45, center=True)

    # ------------------------------------------------------------------
    # Paused
    # ------------------------------------------------------------------

    def draw_paused(self, canvas: np.ndarray):
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
        self._text(canvas, "PAUSED", self.w // 2, self.h // 2,
                   scale=2.5, color=self.WHITE, center=True, thickness=3)
        self._text(canvas, "Press P to resume", self.w // 2, self.h // 2 + 60,
                   scale=0.7, center=True)

    # ------------------------------------------------------------------
    # Finished
    # ------------------------------------------------------------------

    def draw_finished(
        self,
        canvas: np.ndarray,
        winner: Optional[str],
        player_best: float,
        ai_best: float,
        player_laps: List[float],
        ai_laps: List[float],
    ):
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        self._panel(canvas, self.w // 2 - 280, self.h // 2 - 200, 560, 400)

        color = self.ACCENT if winner == "YOU" else self.AI_COLOR
        self._text(canvas, f"🏆  {winner} WINS!", self.w // 2, self.h // 2 - 145,
                   scale=1.8, color=color, center=True, thickness=2)

        self._divider(canvas, self.h // 2 - 105)

        # Stats table
        rows = [
            ("", "YOU", "AI"),
            ("Best Lap", _fmt_time(player_best), _fmt_time(ai_best)),
        ]
        for i, (lp_p, lp_ai) in enumerate(zip(player_laps, ai_laps)):
            rows.append((f"Lap {i+1}", _fmt_time(lp_p), _fmt_time(lp_ai)))

        for r, row in enumerate(rows):
            y = self.h // 2 - 75 + r * 38
            cols = [(self.w // 2 - 200, row[0]),
                    (self.w // 2 + 20, row[1]),
                    (self.w // 2 + 160, row[2])]
            for cx, txt in cols:
                c = self.WHITE if r > 0 else self.ACCENT
                self._text(canvas, txt, cx, y, scale=0.65, color=c)

        self._divider(canvas, self.h // 2 + 155)
        self._text(canvas, "Press R to restart  |  Q to quit",
                   self.w // 2, self.h // 2 + 180, scale=0.6, center=True)

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _panel(self, canvas: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.65):
        sub = canvas[y:y + h, x:x + w]
        bg = np.full_like(sub, 10)
        cv2.addWeighted(bg, alpha, sub, 1 - alpha, 0, sub)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 60), 1)

    def _text(self, canvas, text, x, y, scale=0.7, color=None, center=False, thickness=1):
        color = color or self.WHITE
        font = cv2.FONT_HERSHEY_SIMPLEX
        if center:
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x -= tw // 2
            y -= th // 2
        cv2.putText(canvas, text, (int(x), int(y)), font, scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (int(x), int(y)), font, scale, color,
                    thickness, cv2.LINE_AA)

    def _divider(self, canvas: np.ndarray, y: int):
        cv2.line(canvas, (self.w // 2 - 240, y), (self.w // 2 + 240, y),
                 (60, 60, 60), 1, cv2.LINE_AA)
