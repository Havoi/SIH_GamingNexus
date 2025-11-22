"""
EMG Dino Runner — corrected, robust single-file game
- Uses a trained RandomForest REST/ACTIVE model (joblib)
- Reads EMG from serial (supports single-value or raw,envelope CSV lines)
- Uses sliding window features identical to training
- Debounce logic: requires N consecutive ACTIVE windows to trigger a jump,
  and requires M consecutive REST windows to consider the gesture ended
- Includes keyboard fallback for testing

NOTE: Your uploaded CSV is available at: /mnt/data/labeled_stream_3class.csv
(If you want the 2-class CSV path, change the constant below.)
"""

import sys
import time
import random
from collections import deque, Counter

import numpy as np
import pygame
import joblib

try:
    import serial
except Exception:
    serial = None

# ---------------- CONFIG ----------------
COM_PORT = "COM5"           # change to your serial port or leave
BAUD_RATE = 115200

MODEL_PATH = "emg_rest_active_rf_model.pkl"
LABEL_ENCODER_PATH = "emg_rest_active_label_encoder.pkl"

# If you want to reference uploaded CSV in code/comments / UI
UPLOADED_CSV_PATH = "/mnt/data/labeled_stream_3class.csv"

# Window settings (must match feature extraction & training)
# Example: fs=500, window_sec=0.2 -> WIN_SIZE=100, STEP_SIZE=25
WIN_SIZE = 100
STEP_SIZE = 25

# Debounce parameters (tweak experimentally)
REQUIRED_ACTIVE_FRAMES = 2   # need this many consecutive ACTIVE predictions to trigger
REQUIRED_REST_FRAMES = 3     # need this many consecutive REST predictions to end gesture

REST_LABEL = "REST"
ACTIVE_LABEL = "ACTIVE"

# Game settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 300
GROUND_Y = 240
PLAYER_WIDTH = 40
PLAYER_HEIGHT = 40
JUMP_VELOCITY = -12
GRAVITY = 0.7
OBSTACLE_WIDTH = 20
OBSTACLE_HEIGHT = 40
OBSTACLE_MIN_GAP = 200
OBSTACLE_MAX_GAP = 400
INITIAL_SPEED = 6
SPEED_INCREMENT = 0.001
FPS = 60
# -----------------------------------------

# === EMG features (must match training) ===

def extract_emg_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    x_centered = x - np.mean(x)
    x_abs = np.abs(x_centered)

    std = np.std(x_centered)
    thr = 0.05 * std if std > 0 else 0.0

    mav = np.mean(x_abs)
    rms = np.sqrt(np.mean(x_centered ** 2))
    var = np.var(x_centered)
    wl = np.sum(np.abs(np.diff(x_centered))) if x_centered.size > 1 else 0.0

    zc = 0
    if x_centered.size > 1:
        zc = np.sum(((x_centered[:-1] * x_centered[1:]) < 0) & (np.abs(np.diff(x_centered)) > thr))

    ssc = 0
    if x_centered.size > 2:
        dx1 = np.diff(x_centered)
        ssc = np.sum(((dx1[:-1] * dx1[1:]) < 0) & (np.abs(dx1[:-1]) > thr) & (np.abs(dx1[1:]) > thr))

    wamp = np.sum(np.abs(np.diff(x_centered)) > thr) if x_centered.size > 1 else 0
    iemg = np.sum(x_abs)

    return np.array([mav, rms, var, wl, zc, ssc, wamp, iemg], dtype=float)

# === Player / Obstacle classes ===
class Player:
    def _init_(self, x, y):
        self.x = float(x)
        self.base_y = float(y)
        self.y = float(y)
        self.width = PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        self.vel_y = 0.0
        self.is_jumping = False

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), int(self.width), int(self.height))

    def update(self):
        self.vel_y += GRAVITY
        self.y += self.vel_y
        if self.y >= self.base_y:
            self.y = self.base_y
            self.vel_y = 0.0
            self.is_jumping = False

    def jump(self):
        if not self.is_jumping and self.y >= self.base_y:
            self.vel_y = JUMP_VELOCITY
            self.is_jumping = True

    def draw(self, surface):
        pygame.draw.rect(surface, (50, 200, 50), self.rect)


class Obstacle:
    def _init_(self, x, y, width, height, speed):
        self.x = float(x)
        self.y = float(y)
        self.width = int(width)
        self.height = int(height)
        self.speed = float(speed)

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)

    def update(self):
        self.x -= self.speed

    def draw(self, surface):
        pygame.draw.rect(surface, (200, 50, 50), self.rect)

    def is_offscreen(self):
        return self.x + self.width < 0

# === Serial & model helpers ===

def init_serial():
    if serial is None:
        print("pyserial not installed — running in keyboard-only debug mode")
        return None
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0)
        time.sleep(2.0)
        print(f"Opened serial {COM_PORT} @ {BAUD_RATE}")
        return ser
    except Exception as e:
        print("Could not open serial port:", e)
        print("Continuing in keyboard-only debug mode.")
        return None


def init_model():
    try:
        model = joblib.load(MODEL_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        print("Loaded model and label encoder. Classes:", le.classes_)
        return model, le
    except Exception as e:
        print("Could not load model/label encoder:", e)
        print("Make sure the files exist. Exiting.")
        sys.exit(1)


def read_emg_and_predict(ser, buffer, sample_counter, model, le, last_prediction):
    """Read serial non-blocking, fill buffer, and possibly predict. Returns (buffer, sample_counter, prediction, sample_counter_changed)"""
    pred = last_prediction
    updated = False

    if ser is None:
        return buffer, sample_counter, pred, updated

    while ser.in_waiting > 0:
        raw = ser.readline().decode('utf-8', errors='ignore').strip()
        if not raw:
            continue
        try:
            if ',' in raw:
                parts = raw.split(',')
                val = float(parts[-1])
            else:
                val = float(raw) if '.' in raw else int(raw)
        except Exception:
            continue

        buffer.append(float(val))
        sample_counter += 1
        updated = True

        if len(buffer) == WIN_SIZE and (sample_counter % STEP_SIZE == 0):
            window = np.array(buffer, dtype=float)
            feat = extract_emg_features(window)
            if feat is not None:
                label_int = model.predict(feat.reshape(1, -1))[0]
                label = le.inverse_transform([label_int])[0]
                pred = label

    return buffer, sample_counter, pred, updated

# === Main game ===

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("EMG Dino Runner — REST/ACTIVE")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    big_font = pygame.font.SysFont(None, 48)

    model, le = init_model()
    ser = init_serial()

    emg_buffer = deque(maxlen=WIN_SIZE)
    sample_counter = 0
    last_model_label = REST_LABEL

    # Debounce state
    active_count = 0
    rest_count = 0
    stable_state = REST_LABEL

    # Game state
    player = Player(50, GROUND_Y - PLAYER_HEIGHT)
    obstacles = []
    game_speed = INITIAL_SPEED
    score = 0.0
    game_over = False
    start_time = time.time()

    print("Controls: SPACE to jump (keyboard). Press ESC to quit.")
    print("EMG: perform contraction to generate ACTIVE label (must match training).")

    try:
        while True:
            dt = clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt()
                    if event.key == pygame.K_SPACE and not game_over:
                        player.jump()
                    if event.key == pygame.K_r and game_over:
                        # restart
                        player = Player(50, GROUND_Y - PLAYER_HEIGHT)
                        obstacles = []
                        game_speed = INITIAL_SPEED
                        score = 0.0
                        game_over = False
                        emg_buffer.clear()
                        sample_counter = 0
                        last_model_label = REST_LABEL
                        active_count = rest_count = 0
                        stable_state = REST_LABEL

            if not game_over:
                emg_buffer, sample_counter, model_label, updated = read_emg_and_predict(
                    ser, emg_buffer, sample_counter, model, le, last_model_label
                )

                # Debounce logic: require consecutive frames
                if model_label == ACTIVE_LABEL:
                    active_count += 1
                    rest_count = 0
                else:
                    rest_count += 1
                    active_count = 0

                # Determine stable_state transitions
                if stable_state == REST_LABEL:
                    if active_count >= REQUIRED_ACTIVE_FRAMES:
                        stable_state = ACTIVE_LABEL
                        # trigger jump on rising edge
                        player.jump()
                else:  # currently ACTIVE
                    if rest_count >= REQUIRED_REST_FRAMES:
                        stable_state = REST_LABEL

                last_model_label = model_label

                # Update game physics
                player.update()

                # spawn obstacles
                if not obstacles:
                    nx = SCREEN_WIDTH + random.randint(OBSTACLE_MIN_GAP, OBSTACLE_MAX_GAP)
                    obstacles.append(Obstacle(nx, GROUND_Y - OBSTACLE_HEIGHT, OBSTACLE_WIDTH, OBSTACLE_HEIGHT, game_speed))
                else:
                    # spawn new obstacle based on last obstacle x
                    if obstacles[-1].x < SCREEN_WIDTH - random.randint(OBSTACLE_MIN_GAP, OBSTACLE_MAX_GAP):
                        nx = SCREEN_WIDTH + random.randint(0, 50)
                        obstacles.append(Obstacle(nx, GROUND_Y - OBSTACLE_HEIGHT, OBSTACLE_WIDTH, OBSTACLE_HEIGHT, game_speed))

                # update obstacles
                for obs in obstacles:
                    obs.speed = game_speed
                    obs.update()

                obstacles = [o for o in obstacles if not o.is_offscreen()]

                # collisions
                for obs in obstacles:
                    if player.rect.colliderect(obs.rect):
                        game_over = True
                        break

                # score & speed
                if not game_over:
                    score += dt * 10
                    game_speed += SPEED_INCREMENT

            # Drawing
            screen.fill((240, 240, 240))
            pygame.draw.line(screen, (150, 150, 150), (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)
            player.draw(screen)
            for obs in obstacles:
                obs.draw(screen)

            score_surf = font.render(f"Score: {int(score)}", True, (0, 0, 0))
            label_surf = font.render(f"State: {stable_state}", True, (0, 0, 0))
            csv_surf = font.render(f"CSV: {UPLOADED_CSV_PATH}", True, (80, 80, 80))
            screen.blit(score_surf, (10, 10))
            screen.blit(label_surf, (10, 30))
            screen.blit(csv_surf, (10, 50))

            if game_over:
                over = big_font.render("GAME OVER", True, (200, 0, 0))
                info = font.render("Press R to restart, ESC to quit", True, (0, 0, 0))
                screen.blit(over, (SCREEN_WIDTH // 2 - over.get_width() // 2, SCREEN_HEIGHT // 2 - 40))
                screen.blit(info, (SCREEN_WIDTH // 2 - info.get_width() // 2, SCREEN_HEIGHT // 2 + 10))

            pygame.display.flip()

    except KeyboardInterrupt:
        pass

    finally:
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        pygame.quit()
        print("Exiting.")


if __name__ == '__main__':
    main()