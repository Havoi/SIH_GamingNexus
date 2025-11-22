import pygame
import serial
import time
from collections import deque
import numpy as np
import joblib

# ================== USER CONFIG =====================
COM_PORT = "COM5"            # change to your port (e.g. "COM3", "/dev/ttyUSB0")
BAUD_RATE = 115200

MODEL_PATH = "emg_wrist_rf_model.pkl"      # or your LR model
LABEL_ENCODER_PATH = "emg_label_encoder.pkl"

# Window settings (MUST match how you trained features!)
# Example: if you trained with ~5 Hz and 1s windows -> win_size ~ 4-5
WIN_SIZE = 20      # number of samples per EMG window
STEP_SIZE = 1.5   # new prediction every STEP_SIZE samples

# Game settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
GROUND_Y = 320

PLAYER_X = 100
PLAYER_SIZE = 40
JUMP_VELOCITY = -14
GRAVITY = 0.8

# How long to keep "ACTIVE" after a detection (in frames)
ACTIVE_HOLD_FRAMES = 10
# ====================================================


# === EMG feature extraction (same as training) ===
def extract_emg_features(x: np.ndarray) -> np.ndarray:
    """
    x: 1D numpy array of EMG samples in a window.
    Returns: feature vector [MAV, RMS, VAR, WL, ZC, SSC, WAMP, IEMG]
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0:
        return None

    # center and rectify
    x_centered = x - np.mean(x)
    x_abs = np.abs(x_centered)

    std = np.std(x_centered)
    thr = 0.05 * std if std > 0 else 0.0

    mav = np.mean(x_abs)
    rms = np.sqrt(np.mean(x_centered ** 2))
    var = np.var(x_centered)
    wl = np.sum(np.abs(np.diff(x_centered)))

    # zero crossings
    zc = np.sum(
        ((x_centered[:-1] * x_centered[1:]) < 0) &
        (np.abs(np.diff(x_centered)) > thr)
    )

    # slope sign changes
    dx1 = np.diff(x_centered)
    ssc = np.sum(
        ((dx1[:-1] * dx1[1:]) < 0) &
        (np.abs(dx1[:-1]) > thr) &
        (np.abs(dx1[1:]) > thr)
    )

    # Willison amplitude
    wamp = np.sum(np.abs(np.diff(x_centered)) > thr)

    # Integrated EMG
    iemg = np.sum(x_abs)

    return np.array([mav, rms, var, wl, zc, ssc, wamp, iemg], dtype=float)


def init_serial():
    print(f"Opening serial port {COM_PORT} @ {BAUD_RATE}...")
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2.0)  # allow Arduino reset
    return ser


def init_model():
    print("Loading model and label encoder...")
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    print("Classes:", le.classes_)
    return model, le


def main():
    # ---- Init model + serial ----
    model, le = init_model()
    ser = init_serial()

    # ---- Init pygame ----
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("EMG Wrist Runner (REST vs NON-REST)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)

    # Player state
    player_x = PLAYER_X
    player_y = GROUND_Y - PLAYER_SIZE
    player_vel_y = 0.0

    # EMG state
    emg_buffer = deque(maxlen=WIN_SIZE)
    sample_counter = 0
    current_label = "REST"
    active_frames_left = 0  # frames to keep "ACTIVE"

    running = True
    print("✅ Game started. Use your wrist EMG to JUMP (UP or DOWN = ACTIVE).")
    print("Press ESC or close the window to quit.")

    try:
        while running:
            # ====== Handle events ======
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # ====== Read EMG from serial, update state ======
            while ser.in_waiting > 0:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw_line:
                    continue

                # parse to number
                try:
                    if "." in raw_line:
                        val = float(raw_line)
                    else:
                        val = int(raw_line)
                except ValueError:
                    continue

                emg_buffer.append(float(val))
                sample_counter += 1

                if len(emg_buffer) == WIN_SIZE and (sample_counter % STEP_SIZE == 0):
                    window = np.array(emg_buffer, dtype=float)
                    feat_vec = extract_emg_features(window)
                    if feat_vec is None:
                        continue

                    feat_vec_2d = feat_vec.reshape(1, -1)
                    y_pred_int = model.predict(feat_vec_2d)[0]
                    label = le.inverse_transform([y_pred_int])[0]

                    # REST vs NON-REST mapping:
                    # if label is "REST" -> REST
                    # if label is "UP" or "DOWN" -> ACTIVE
                    if label == "REST":
                        current_label = "REST"
                    else:
                        current_label = "ACTIVE"
                        active_frames_left = ACTIVE_HOLD_FRAMES  # hold a bit

                    # Debug print (optional)
                    # print("Raw label:", label, "Mapped:", current_label)

            # Decrease active hold
            if active_frames_left > 0:
                active_frames_left -= 1
                # keep ACTIVE while counter > 0
                if current_label != "ACTIVE":
                    current_label = "ACTIVE"
            else:
                # if hold expired and last label wasn't ACTIVE from EMG
                if current_label == "ACTIVE":
                    current_label = "REST"

            is_active = (current_label == "ACTIVE")

            # ====== Game logic: JUMP on ACTIVE ======
            # allow jump only if on ground
            on_ground = (player_y >= GROUND_Y - PLAYER_SIZE - 0.1)

            if is_active and on_ground:
                player_vel_y = JUMP_VELOCITY

            # Apply gravity
            player_vel_y += GRAVITY
            player_y += player_vel_y

            # Clamp to ground
            if player_y >= GROUND_Y - PLAYER_SIZE:
                player_y = GROUND_Y - PLAYER_SIZE
                player_vel_y = 0.0

            # ====== Drawing ======
            screen.fill((30, 30, 30))

            # Ground
            pygame.draw.line(screen, (200, 200, 200),
                             (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)

            # Player
            player_rect = pygame.Rect(int(player_x), int(player_y),
                                      PLAYER_SIZE, PLAYER_SIZE)
            pygame.draw.rect(screen, (100, 200, 250), player_rect)

            # Status text
            status_text = f"State: {current_label} (REST vs NON-REST)"
            text_surface = font.render(status_text, True, (255, 255, 255))
            screen.blit(text_surface, (20, 20))

            info_text = "Jump when you contract (UP or DOWN). Relax = REST."
            info_surface = font.render(info_text, True, (180, 180, 180))
            screen.blit(info_surface, (20, 60))

            pygame.display.flip()
            clock.tick(60)  # 60 FPS

    finally:
        ser.close()
        pygame.quit()
        print("Game stopped, serial closed.")


if __name__ == "__main__":
    main()