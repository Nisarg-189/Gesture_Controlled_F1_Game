"""
GripAI — AI Opponent
An adaptive racing AI that:
  1. Follows the track using a lookahead pure-pursuit controller
  2. Observes the player's driving style (aggression, braking points, cornering speed)
  3. Gradually mirrors and adapts to match / challenge the player

Adaptation model:
  - Tracks a rolling window of player throttle / brake / speed telemetry
  - Computes a "player profile" (aggression score, preferred cornering speed)
  - Blends its own base behaviour toward the player profile as adaptation_level grows
"""

import math
import random
import collections
import numpy as np
from typing import Tuple, Optional

from src.utils.logger import setup_logger

logger = setup_logger("AIOpponent")

# Type hints (avoid circular import)
if False:  # TYPE_CHECKING only
    from src.game.car import Car
    from src.game.track import Track


class DrivingProfile:
    """Statistical model of a driver's style, built from telemetry."""

    WINDOW = 600   # samples (~10 s at 60 fps)

    def __init__(self):
        self._throttle = collections.deque(maxlen=self.WINDOW)
        self._brake = collections.deque(maxlen=self.WINDOW)
        self._speed = collections.deque(maxlen=self.WINDOW)
        self._steer = collections.deque(maxlen=self.WINDOW)

    def record(self, throttle: float, brake: float, steer: float, speed: float):
        self._throttle.append(throttle)
        self._brake.append(brake)
        self._steer.append(steer)
        self._speed.append(speed)

    @property
    def aggression(self) -> float:
        """0 = timid, 1 = flat-out. Based on avg throttle minus brake."""
        if not self._throttle:
            return 0.5
        t = np.mean(self._throttle)
        b = np.mean(self._brake)
        return float(np.clip(t - b * 0.5 + 0.4, 0.0, 1.0))

    @property
    def avg_corner_speed(self) -> float:
        """Average speed when steering hard (|steer| > 0.4)."""
        speeds = [s for s, st in zip(self._speed, self._steer) if abs(st) > 0.4]
        return float(np.mean(speeds)) if speeds else 200.0

    @property
    def braking_sharpness(self) -> float:
        """How hard the player brakes on average."""
        if not self._brake:
            return 0.5
        return float(np.clip(np.mean(self._brake) * 3.0, 0.0, 1.0))

    @property
    def sample_count(self) -> int:
        return len(self._throttle)


class AIOpponent:
    """
    Pure-pursuit path follower + adaptive style mirror.
    """

    DIFFICULTY_PROFILES = {
        "easy":     {"speed_mult": 0.72, "adapt_rate": 0.0002},
        "medium":   {"speed_mult": 0.88, "adapt_rate": 0.0005},
        "hard":     {"speed_mult": 1.00, "adapt_rate": 0.0008},
        "adaptive": {"speed_mult": 0.80, "adapt_rate": 0.001},
    }

    # Pure pursuit
    LOOKAHEAD_BASE = 60.0    # pixels ahead to aim for
    LOOKAHEAD_SPEED_SCALE = 0.25

    # Driving parameters (will be adapted)
    BASE_TARGET_SPEED = 260.0    # pixels/s
    BASE_CORNER_SLOWDOWN = 0.55
    BASE_BRAKE_DISTANCE = 80.0

    MAX_ADAPT_LEVEL = 1.0
    ADAPT_RAMP_SAMPLES = 3600   # ~60 s to reach full adaptation

    def __init__(self, difficulty: str = "adaptive"):
        self.difficulty = difficulty
        cfg = self.DIFFICULTY_PROFILES.get(difficulty, self.DIFFICULTY_PROFILES["adaptive"])
        self._speed_mult = cfg["speed_mult"]
        self._adapt_rate = cfg["adapt_rate"]

        self._player_profile = DrivingProfile()
        self.adaptation_level = 0.0    # 0..1, shown on HUD

        # Internal AI state
        self._target_speed = self.BASE_TARGET_SPEED * self._speed_mult
        self._corner_slowdown = self.BASE_CORNER_SLOWDOWN
        self._brake_distance = self.BASE_BRAKE_DISTANCE

        # Error integral for PID-like steer
        self._steer_integral = 0.0
        self._prev_steer_err = 0.0

        logger.info("AIOpponent initialised: difficulty=%s speed_mult=%.2f", difficulty, self._speed_mult)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe_player(self, throttle: float, brake: float, steer: float, speed: float):
        """Call every tick to record player telemetry."""
        self._player_profile.record(throttle, brake, steer, speed)
        # Grow adaptation level
        samples = self._player_profile.sample_count
        self.adaptation_level = min(self.MAX_ADAPT_LEVEL, samples / self.ADAPT_RAMP_SAMPLES)
        # Update driving params from player profile
        self._adapt_driving_style()

    def decide(self, ai_car, player_car, track, dt: float) -> Tuple[float, float, float]:
        """
        Compute (throttle, brake, steer) for the AI car.
        """
        # Path ahead
        lookahead_dist = self.LOOKAHEAD_BASE + ai_car.speed * self.LOOKAHEAD_SPEED_SCALE
        path_pts = track.get_path_ahead(ai_car.x, ai_car.y, n=10)
        if not path_pts:
            return 0.5, 0.0, 0.0

        # Find lookahead point
        target = self._find_lookahead_pt(ai_car, path_pts, lookahead_dist)

        # Steer toward target (pure pursuit)
        steer = self._pure_pursuit_steer(ai_car, target, dt)

        # Speed control
        curvature = self._estimate_curvature(path_pts)
        throttle, brake = self._speed_control(ai_car, curvature, player_car)

        # Micro-randomness to avoid robot-perfect driving
        noise = random.gauss(0, 0.02 * (1.0 - self.adaptation_level * 0.5))
        steer = float(np.clip(steer + noise, -1.0, 1.0))

        return float(throttle), float(brake), float(steer)

    def reset(self):
        self._player_profile = DrivingProfile()
        self.adaptation_level = 0.0
        self._target_speed = self.BASE_TARGET_SPEED * self._speed_mult
        self._corner_slowdown = self.BASE_CORNER_SLOWDOWN
        self._steer_integral = 0.0
        self._prev_steer_err = 0.0

    # ------------------------------------------------------------------
    # Internal: adaptation
    # ------------------------------------------------------------------

    def _adapt_driving_style(self):
        if self.adaptation_level < 0.05:
            return

        a = self.adaptation_level * self._adapt_rate * 100  # blend weight
        p = self._player_profile

        # Mirror player aggression → target speed
        player_speed_est = p.avg_corner_speed * (0.7 + p.aggression * 0.6)
        self._target_speed = (
            (1 - a) * self._target_speed
            + a * player_speed_est * self._speed_mult
        )
        self._target_speed = float(np.clip(self._target_speed, 120, 400))

        # Mirror braking sharpness → how early the AI brakes
        self._brake_distance = (
            (1 - a) * self._brake_distance
            + a * (self.BASE_BRAKE_DISTANCE * (1.0 + p.braking_sharpness))
        )

        # Mirror corner speed
        self._corner_slowdown = (
            (1 - a) * self._corner_slowdown
            + a * (0.45 + (1.0 - p.aggression) * 0.3)
        )

    # ------------------------------------------------------------------
    # Internal: pure pursuit
    # ------------------------------------------------------------------

    def _find_lookahead_pt(self, ai_car, path_pts, dist: float):
        best = path_pts[-1]
        for pt in path_pts:
            d = math.hypot(pt[0] - ai_car.x, pt[1] - ai_car.y)
            if d >= dist:
                return pt
        return best

    def _pure_pursuit_steer(self, ai_car, target, dt: float) -> float:
        dx = target[0] - ai_car.x
        dy = target[1] - ai_car.y
        target_angle = math.atan2(dy, dx)
        heading_err = target_angle - ai_car.angle
        # Normalise to -π..π
        heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi

        # PD control
        Kp = 1.8
        Kd = 0.4
        d_err = (heading_err - self._prev_steer_err) / max(dt, 1e-4)
        steer = Kp * heading_err + Kd * d_err
        self._prev_steer_err = heading_err
        return float(np.clip(steer / math.pi, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Internal: speed control
    # ------------------------------------------------------------------

    def _estimate_curvature(self, path_pts) -> float:
        if len(path_pts) < 3:
            return 0.0
        angles = []
        for i in range(1, len(path_pts) - 1):
            a = math.atan2(path_pts[i][1] - path_pts[i-1][1],
                           path_pts[i][0] - path_pts[i-1][0])
            b = math.atan2(path_pts[i+1][1] - path_pts[i][1],
                           path_pts[i+1][0] - path_pts[i][0])
            diff = abs((b - a + math.pi) % (2 * math.pi) - math.pi)
            angles.append(diff)
        return float(np.mean(angles))   # higher = tighter corner

    def _speed_control(self, ai_car, curvature: float, player_car) -> Tuple[float, float]:
        # Slow for corners
        corner_factor = max(self._corner_slowdown, 1.0 - curvature * 4.0)
        target = self._target_speed * corner_factor

        # Competitive awareness: if player is far ahead, push harder
        dist_to_player = math.hypot(ai_car.x - player_car.x, ai_car.y - player_car.y)
        if dist_to_player > 100:
            target *= min(1.15, 1.0 + dist_to_player / 1000.0)

        speed = ai_car.speed
        delta = target - speed

        if delta > 10:
            throttle = min(1.0, delta / 50.0 + 0.4)
            brake = 0.0
        elif delta < -20:
            throttle = 0.0
            brake = min(1.0, -delta / 80.0)
        else:
            throttle = 0.3
            brake = 0.0

        # Hard braking for very tight corners
        if curvature > 0.25 and speed > target * 1.1:
            brake = max(brake, 0.6)
            throttle = 0.0

        return throttle, brake
