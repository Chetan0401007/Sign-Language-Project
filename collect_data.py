import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import time

DATA_FILE        = "data.csv"
FRAMES_PER_CLASS = 300
CAMERA_INDEX     = 0
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE  = 0.5

NUM_LANDMARKS = 21
FEATURE_COLS  = [f"x{i}" for i in range(NUM_LANDMARKS)] + \
                [f"y{i}" for i in range(NUM_LANDMARKS)] + \
                [f"z{i}" for i in range(NUM_LANDMARKS)]

mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

def extract_and_normalise(hand_landmarks):
    lm = hand_landmarks.landmark

    xs = np.array([p.x for p in lm])
    ys = np.array([p.y for p in lm])
    zs = np.array([p.z for p in lm])

    xs -= xs[0]
    ys -= ys[0]
    zs -= zs[0]

    scale = np.sqrt((xs.max() - xs.min())**2 + (ys.max() - ys.min())**2) + 1e-6
    xs /= scale
    ys /= scale
    zs /= scale

    return np.concatenate([xs, ys, zs])

def draw_rounded_rect(img, x1, y1, x2, y2, radius, color, thickness=-1):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

def put_text_shadow(img, text, org, font, scale, color, thickness):
    cv2.putText(img, text, (org[0]+2, org[1]+2), font, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording        = False
    current_label    = None
    collected        = 0
    session_buffer   = []

    prev_time = time.time()

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as hands:

        print("=" * 60)
        print("  Sign Language Data Collector  |  Press Q to quit")
        print("  Press any letter (A-Z) to record class")
        print("=" * 60)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            curr_time = time.time()
            fps       = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results    = hands.process(rgb)
            rgb.flags.writeable = True

            hand_detected = False
            features      = None

            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style())

                    features = extract_and_normalise(hand_lm)

                if recording and features is not None:
                    row = list(features) + [current_label]
                    session_buffer.append(row)
                    collected += 1

                    if collected >= FRAMES_PER_CLASS:
                        _save_buffer(session_buffer)
                        session_buffer.clear()
                        print(f"[✓] Saved {FRAMES_PER_CLASS} samples for class '{current_label}'")
                        recording     = False
                        current_label = None
                        collected     = 0

            draw_rounded_rect(frame, 10, 10, w - 10, 90, 12,
                              (20, 20, 40))
            put_text_shadow(frame, f"FPS: {fps:.1f}", (25, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 220, 180), 2)
            hand_str = "Hand: DETECTED" if hand_detected else "Hand: NOT FOUND"
            hand_col = (80, 220, 100) if hand_detected else (60, 60, 200)
            put_text_shadow(frame, hand_str, (220, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, hand_col, 2)

            if recording:
                progress = int((collected / FRAMES_PER_CLASS) * (w - 80))
                cv2.rectangle(frame, (40, h - 60), (w - 40, h - 30),
                              (40, 40, 60), -1)
                cv2.rectangle(frame, (40, h - 60), (40 + progress, h - 30),
                              (0, 200, 120), -1)
                bar_text = f"Recording '{current_label}' — {collected}/{FRAMES_PER_CLASS}"
                put_text_shadow(frame, bar_text, (45, h - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(frame, (w - 55, h - 45), 10, (0, 0, 230), -1)
                    put_text_shadow(frame, "REC", (w - 95, h - 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 1)
            else:
                put_text_shadow(frame,
                                "Press a key (A-Z) to start recording a class",
                                (25, h - 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (180, 180, 220), 2)

            cv2.imshow("Sign Language — Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                if session_buffer:
                    _save_buffer(session_buffer)
                    print(f"[✓] Saved {len(session_buffer)} partial samples.")
                break
            elif key != 255:
                ch = chr(key).upper()
                if ch.isalpha():
                    if recording and session_buffer:
                        _save_buffer(session_buffer)
                        session_buffer.clear()
                        print(f"[!] Switching class — saved {collected} partial samples.")
                    current_label = ch
                    collected     = 0
                    recording     = True
                    print(f"[●] Recording class: '{current_label}' — show your sign!")

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Data collection complete. Check data.csv")


def _save_buffer(buffer: list):
    df    = pd.DataFrame(buffer, columns=FEATURE_COLS + ["label"])
    write_header = not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0
    df.to_csv(DATA_FILE, mode="a", header=write_header, index=False)


if __name__ == "__main__":
    main()
