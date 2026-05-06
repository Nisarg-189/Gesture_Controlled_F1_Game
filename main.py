"""
GripAI — Gesture-Controlled F1 Mini Game
Entry point: launches the game loop with webcam gesture control and AI opponent.
"""

import sys
import argparse
import cv2
from src.game.engine import GameEngine
from src.vision.gesture_controller import GestureController
from src.ai.opponent import AIOpponent
from src.utils.logger import setup_logger

logger = setup_logger("GripAI")


def parse_args():
    parser = argparse.ArgumentParser(
        description="GripAI — Control an F1 car with hand gestures. Race an AI that learns you."
    )
    parser.add_argument("--width", type=int, default=1280, help="Game window width")
    parser.add_argument("--height", type=int, default=720, help="Game window height")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    parser.add_argument("--ai-difficulty", choices=["easy", "medium", "hard", "adaptive"],
                        default="adaptive", help="AI difficulty mode")
    parser.add_argument("--no-gesture", action="store_true",
                        help="Disable gesture control (use keyboard instead)")
    parser.add_argument("--debug", action="store_true", help="Show debug overlays")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("🏎️  GripAI starting up...")

    # Initialise gesture controller
    gesture_ctrl = None
    if not args.no_gesture:
        logger.info("📷  Initialising gesture controller (camera index %d)...", args.camera)
        gesture_ctrl = GestureController(camera_index=args.camera, debug=args.debug)
        if not gesture_ctrl.is_available():
            logger.warning("⚠️  Camera not available — falling back to keyboard control.")
            gesture_ctrl = None

    # Initialise AI opponent
    logger.info("🤖  Initialising AI opponent (mode: %s)...", args.ai_difficulty)
    ai_opponent = AIOpponent(difficulty=args.ai_difficulty)

    # Launch game engine
    logger.info("🏁  Launching game engine (%dx%d @ %d FPS)...", args.width, args.height, args.fps)
    engine = GameEngine(
        width=args.width,
        height=args.height,
        fps=args.fps,
        gesture_controller=gesture_ctrl,
        ai_opponent=ai_opponent,
        debug=args.debug,
    )

    try:
        engine.run()
    except KeyboardInterrupt:
        logger.info("🛑  Interrupted by user.")
    finally:
        if gesture_ctrl:
            gesture_ctrl.release()
        cv2.destroyAllWindows()
        logger.info("✅  GripAI shut down cleanly.")


if __name__ == "__main__":
    main()
