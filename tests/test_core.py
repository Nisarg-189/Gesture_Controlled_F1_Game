"""
GripAI — Unit Tests
Tests for physics engine, AI driving profile, and gesture input parsing.
"""

import math
import unittest
from unittest.mock import MagicMock


class TestPhysicsEngine(unittest.TestCase):

    def setUp(self):
        from src.game.physics import PhysicsEngine
        from src.game.car import Car
        self.engine = PhysicsEngine()
        self.car = Car(start_pos=(100, 100), start_angle=0.0,
                       color=(255, 0, 0), label="TEST")

    def test_throttle_accelerates_car(self):
        initial_speed = self.car.speed
        for _ in range(30):
            self.engine.update(self.car, throttle=1.0, brake=0.0, steer=0.0, dt=1/60)
        self.assertGreater(self.car.speed, initial_speed + 10)

    def test_brake_decelerates_car(self):
        # First accelerate
        for _ in range(60):
            self.engine.update(self.car, throttle=1.0, brake=0.0, steer=0.0, dt=1/60)
        speed_before = self.car.speed
        # Then brake hard
        for _ in range(30):
            self.engine.update(self.car, throttle=0.0, brake=1.0, steer=0.0, dt=1/60)
        self.assertLess(self.car.speed, speed_before * 0.6)

    def test_speed_cap_respected(self):
        from src.game.physics import PhysicsEngine
        for _ in range(300):
            self.engine.update(self.car, throttle=1.0, brake=0.0, steer=0.0, dt=1/60)
        self.assertLessEqual(self.car.speed, PhysicsEngine.MAX_SPEED + 1.0)

    def test_steering_changes_angle(self):
        initial_angle = self.car.angle
        for _ in range(60):
            self.engine.update(self.car, throttle=0.8, brake=0.0, steer=1.0, dt=1/60)
        self.assertNotAlmostEqual(self.car.angle, initial_angle, places=2)

    def test_no_input_decelerates(self):
        for _ in range(60):
            self.engine.update(self.car, throttle=1.0, brake=0.0, steer=0.0, dt=1/60)
        speed_peak = self.car.speed
        for _ in range(120):
            self.engine.update(self.car, throttle=0.0, brake=0.0, steer=0.0, dt=1/60)
        self.assertLess(self.car.speed, speed_peak)


class TestDrivingProfile(unittest.TestCase):

    def setUp(self):
        from src.ai.opponent import DrivingProfile
        self.profile = DrivingProfile()

    def test_aggression_flat_out(self):
        for _ in range(200):
            self.profile.record(throttle=1.0, brake=0.0, steer=0.0, speed=300)
        self.assertGreater(self.profile.aggression, 0.7)

    def test_aggression_heavy_braker(self):
        for _ in range(200):
            self.profile.record(throttle=0.3, brake=0.8, steer=0.0, speed=100)
        self.assertLess(self.profile.aggression, 0.5)

    def test_avg_corner_speed(self):
        for _ in range(100):
            self.profile.record(throttle=0.5, brake=0.0, steer=0.6, speed=180)
        self.assertAlmostEqual(self.profile.avg_corner_speed, 180, delta=5)

    def test_sample_count_grows(self):
        for i in range(50):
            self.profile.record(0.5, 0.1, 0.2, 200)
        self.assertEqual(self.profile.sample_count, 50)


class TestAIOpponent(unittest.TestCase):

    def _make_car(self, x=640, y=360, speed=200):
        from src.game.car import Car
        car = Car(start_pos=(x, y), start_angle=0.0, color=(0, 255, 0), label="AI")
        car.speed = speed
        car.vx = speed
        car.vy = 0.0
        return car

    def _make_track(self):
        track = MagicMock()
        track.get_path_ahead.return_value = [
            (700, 360), (760, 355), (820, 345), (880, 330),
            (930, 310), (970, 285), (1000, 255), (1020, 220),
        ]
        return track

    def test_decide_returns_valid_range(self):
        from src.ai.opponent import AIOpponent
        ai = AIOpponent(difficulty="medium")
        ai_car = self._make_car()
        player_car = self._make_car(x=700)
        track = self._make_track()
        throttle, brake, steer = ai.decide(ai_car, player_car, track, dt=1/60)
        self.assertGreaterEqual(throttle, 0.0)
        self.assertLessEqual(throttle, 1.0)
        self.assertGreaterEqual(brake, 0.0)
        self.assertLessEqual(brake, 1.0)
        self.assertGreaterEqual(steer, -1.0)
        self.assertLessEqual(steer, 1.0)

    def test_adaptation_level_grows(self):
        from src.ai.opponent import AIOpponent
        ai = AIOpponent(difficulty="adaptive")
        for _ in range(500):
            ai.observe_player(0.8, 0.1, 0.3, 250)
        self.assertGreater(ai.adaptation_level, 0.0)

    def test_reset_clears_state(self):
        from src.ai.opponent import AIOpponent
        ai = AIOpponent(difficulty="adaptive")
        for _ in range(500):
            ai.observe_player(0.8, 0.1, 0.3, 250)
        ai.reset()
        self.assertEqual(ai.adaptation_level, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
