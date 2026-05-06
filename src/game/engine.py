"""
GripAI Game Engine
Core game loop, rendering pipeline, and state management.
"""

import time
import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.game.track import Track
from src.game.car import Car, CarState
from src.game.physics import PhysicsEngine
from src.game.hud import HUD
from src.vision.gesture_controller import GestureController, GestureInput
from src.ai.opponent import AIOpponent
from src.utils.logger import setup_logger

logger = setup_logger("GameEngine")


class GameState(Enum):
    MENU = auto()
    COUNTDOWN = auto()
    RACING = auto()
    PAUSED = auto()
    FINISHED = auto()


@dataclass
class RaceStats:
    lap: int = 1
    total_laps: int = 3
    player_lap_times: list = field(default_factory=list)
    ai_lap_times: list = field(default_factory=list)
    player_best_lap: float = float("inf")
    ai_best_lap: float = float("inf")
    race_start_time: float = 0.0
    lap_start_time: float = 0.0
    player_finished: bool = False
    ai_finished: bool = False
    winner: Optional[str] = None


class GameEngine:
    """
    Main game engine. Coordinates physics, rendering, gesture input, and AI.
    """

    TARGET_W = 1280
    TARGET_H = 720
    COUNTDOWN_SECS = 3

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        gesture_controller: Optional[GestureController] = None,
        ai_opponent: Optional[AIOpponent] = None,
        debug: bool = False,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.dt = 1.0 / fps
        self.gesture_ctrl = gesture_controller
        self.ai = ai_opponent
        self.debug = debug

        # Build game objects
        self.track = Track(width, height)
        self.player_car = Car(
            start_pos=self.track.start_position,
            start_angle=self.track.start_angle,
            color=(220, 50, 50),   # Red — player
            label="YOU",
        )
        self.ai_car = Car(
            start_pos=self.track.ai_start_position,
            start_angle=self.track.start_angle,
            color=(50, 180, 255),  # Cyan — AI
            label="AI",
        )
        self.physics = PhysicsEngine()
        self.hud = HUD(width, height)

        self.state = GameState.MENU
        self.race_stats = RaceStats()
        self.countdown_value = self.COUNTDOWN_SECS
        self._countdown_start = 0.0

        # Canvas
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Gesture / keyboard fallback state
        self._kb_throttle = 0.0
        self._kb_brake = 0.0
        self._kb_steer = 0.0

        # Timing
        self._last_time = time.perf_counter()
        self._frame_times = []

        logger.info("GameEngine initialised. Track: %s", self.track.name)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        cv2.namedWindow("GripAI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GripAI", self.width, self.height)

        while True:
            now = time.perf_counter()
            dt = min(now - self._last_time, 0.05)  # cap at 50 ms to avoid spiral
            self._last_time = now

            # --- Input ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            self._handle_keyboard(key)

            gesture: Optional[GestureInput] = None
            if self.gesture_ctrl:
                gesture = self.gesture_ctrl.get_input()

            # --- Update ---
            self._update(dt, gesture, key)

            # --- Render ---
            self._render(dt)

            # --- Display ---
            cv2.imshow("GripAI", self.canvas)

            # FPS limiter
            elapsed = time.perf_counter() - now
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # State machine update
    # ------------------------------------------------------------------

    def _update(self, dt: float, gesture: Optional["GestureInput"], key: int):
        if self.state == GameState.MENU:
            if key == ord("\r") or key == 13 or key == ord(" "):
                self._start_countdown()

        elif self.state == GameState.COUNTDOWN:
            elapsed = time.perf_counter() - self._countdown_start
            remaining = self.COUNTDOWN_SECS - elapsed
            self.countdown_value = max(0, remaining)
            if remaining <= 0:
                self._start_race()

        elif self.state == GameState.RACING:
            self._update_racing(dt, gesture)

        elif self.state == GameState.PAUSED:
            if key == ord("p"):
                self.state = GameState.RACING

        elif self.state == GameState.FINISHED:
            if key == ord("r"):
                self._reset()

        # Global pause toggle
        if key == ord("p") and self.state == GameState.RACING:
            self.state = GameState.PAUSED

    def _update_racing(self, dt: float, gesture: Optional["GestureInput"]):
        # --- Player input resolution ---
        if gesture:
            throttle = gesture.throttle
            brake = gesture.brake
            steer = gesture.steer
        else:
            throttle = self._kb_throttle
            brake = self._kb_brake
            steer = self._kb_steer

        # --- Physics update: player ---
        self.physics.update(self.player_car, throttle, brake, steer, dt)

        # --- AI decision ---
        if self.ai:
            ai_throttle, ai_brake, ai_steer = self.ai.decide(
                self.ai_car, self.player_car, self.track, dt
            )
        else:
            ai_throttle, ai_brake, ai_steer = 0.8, 0.0, 0.0
        self.physics.update(self.ai_car, ai_throttle, ai_brake, ai_steer, dt)

        # --- Track boundary & collision ---
        self.track.apply_constraints(self.player_car)
        self.track.apply_constraints(self.ai_car)

        # --- Checkpoint / lap detection ---
        self._check_checkpoints(self.player_car, is_player=True)
        self._check_checkpoints(self.ai_car, is_player=False)

        # --- Feed AI style data ---
        if self.ai:
            self.ai.observe_player(throttle, brake, steer, self.player_car.speed)

    # ------------------------------------------------------------------
    # Checkpoint / lap logic
    # ------------------------------------------------------------------

    def _check_checkpoints(self, car: "Car", is_player: bool):
        cp_index = self.track.check_checkpoint(car)
        if cp_index is None:
            return

        if is_player:
            if car.next_checkpoint == cp_index:
                car.next_checkpoint = (cp_index + 1) % len(self.track.checkpoints)
                if cp_index == 0 and car.checkpoints_hit > 0:
                    self._complete_lap(is_player=True)
                car.checkpoints_hit += 1
        else:
            if car.next_checkpoint == cp_index:
                car.next_checkpoint = (cp_index + 1) % len(self.track.checkpoints)
                if cp_index == 0 and car.checkpoints_hit > 0:
                    self._complete_lap(is_player=False)
                car.checkpoints_hit += 1

    def _complete_lap(self, is_player: bool):
        now = time.perf_counter()
        lap_time = now - self.race_stats.lap_start_time
        self.race_stats.lap_start_time = now

        if is_player:
            self.race_stats.player_lap_times.append(lap_time)
            if lap_time < self.race_stats.player_best_lap:
                self.race_stats.player_best_lap = lap_time
            logger.info("Player lap %d: %.3fs", len(self.race_stats.player_lap_times), lap_time)

            if len(self.race_stats.player_lap_times) >= self.race_stats.total_laps:
                self.race_stats.player_finished = True
                if not self.race_stats.winner:
                    self.race_stats.winner = "YOU"
                self._finish_race()
        else:
            self.race_stats.ai_lap_times.append(lap_time)
            if lap_time < self.race_stats.ai_best_lap:
                self.race_stats.ai_best_lap = lap_time
            logger.info("AI lap %d: %.3fs", len(self.race_stats.ai_lap_times), lap_time)

            if len(self.race_stats.ai_lap_times) >= self.race_stats.total_laps:
                self.race_stats.ai_finished = True
                if not self.race_stats.winner:
                    self.race_stats.winner = "AI"
                self._finish_race()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _start_countdown(self):
        self.state = GameState.COUNTDOWN
        self._countdown_start = time.perf_counter()
        self.countdown_value = self.COUNTDOWN_SECS

    def _start_race(self):
        self.state = GameState.RACING
        now = time.perf_counter()
        self.race_stats.race_start_time = now
        self.race_stats.lap_start_time = now

    def _finish_race(self):
        self.state = GameState.FINISHED

    def _reset(self):
        self.player_car = Car(
            start_pos=self.track.start_position,
            start_angle=self.track.start_angle,
            color=(220, 50, 50),
            label="YOU",
        )
        self.ai_car = Car(
            start_pos=self.track.ai_start_position,
            start_angle=self.track.start_angle,
            color=(50, 180, 255),
            label="AI",
        )
        self.race_stats = RaceStats()
        if self.ai:
            self.ai.reset()
        self.state = GameState.MENU

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(self, dt: float):
        # Clear
        self.canvas[:] = (15, 15, 20)

        # Track
        self.track.draw(self.canvas)

        # Cars
        self.ai_car.draw(self.canvas)
        self.player_car.draw(self.canvas)

        # HUD overlay
        if self.state == GameState.MENU:
            self.hud.draw_menu(self.canvas)
        elif self.state == GameState.COUNTDOWN:
            self.hud.draw_countdown(self.canvas, self.countdown_value)
        elif self.state == GameState.RACING:
            now = time.perf_counter()
            race_time = now - self.race_stats.race_start_time
            lap_time = now - self.race_stats.lap_start_time
            player_laps = len(self.race_stats.player_lap_times) + 1
            ai_laps = len(self.race_stats.ai_lap_times) + 1
            self.hud.draw_racing(
                self.canvas,
                player_speed=self.player_car.speed,
                race_time=race_time,
                lap_time=lap_time,
                player_lap=player_laps,
                ai_lap=ai_laps,
                total_laps=self.race_stats.total_laps,
                player_best=self.race_stats.player_best_lap,
                ai_adaptation=self.ai.adaptation_level if self.ai else 0.0,
            )
        elif self.state == GameState.PAUSED:
            self.hud.draw_paused(self.canvas)
        elif self.state == GameState.FINISHED:
            self.hud.draw_finished(
                self.canvas,
                winner=self.race_stats.winner,
                player_best=self.race_stats.player_best_lap,
                ai_best=self.race_stats.ai_best_lap,
                player_laps=self.race_stats.player_lap_times,
                ai_laps=self.race_stats.ai_lap_times,
            )

        # Gesture camera feed (corner PiP)
        if self.gesture_ctrl and self.gesture_ctrl.last_frame is not None:
            self._draw_camera_pip(self.gesture_ctrl.last_frame)

        # Debug overlay
        if self.debug:
            self._draw_debug()

    def _draw_camera_pip(self, frame: np.ndarray):
        pip_w, pip_h = 240, 135
        pip = cv2.resize(frame, (pip_w, pip_h))
        x0, y0 = self.width - pip_w - 10, self.height - pip_h - 10
        # Border
        cv2.rectangle(self.canvas, (x0 - 2, y0 - 2), (x0 + pip_w + 2, y0 + pip_h + 2),
                      (80, 80, 80), 2)
        self.canvas[y0:y0 + pip_h, x0:x0 + pip_w] = pip

    def _draw_debug(self):
        texts = [
            f"Player speed: {self.player_car.speed * 3.6:.1f} km/h",
            f"Player angle: {math.degrees(self.player_car.angle):.1f}°",
            f"AI adaptation: {self.ai.adaptation_level:.2f}" if self.ai else "No AI",
        ]
        for i, t in enumerate(texts):
            cv2.putText(self.canvas, t, (10, 30 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Keyboard fallback
    # ------------------------------------------------------------------

    def _handle_keyboard(self, key: int):
        # Held-key simulation via simple flags toggled by key events
        # Arrow keys / WASD
        if key == ord("w") or key == 82:   # W or Up
            self._kb_throttle = 1.0
        elif key == ord("s") or key == 84:  # S or Down
            self._kb_brake = 1.0
        elif key == ord("a") or key == 81:  # A or Left
            self._kb_steer = -1.0
        elif key == ord("d") or key == 83:  # D or Right
            self._kb_steer = 1.0
        else:
            # Decay when no key pressed
            self._kb_throttle *= 0.85
            self._kb_brake *= 0.85
            self._kb_steer *= 0.80
