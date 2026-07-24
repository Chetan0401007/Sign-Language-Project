import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
from collections import deque, Counter

MODEL_FILE      = "model.joblib"
LABEL_FILE      = "label_classes.joblib"
CAMERA_INDEX    = 0

MIN_DETECTION_CONFIDENCE = 0.75
MIN_TRACKING_CONFIDENCE  = 0.60

SMOOTHING_WINDOW     = 12
CONFIDENCE_THRESHOLD = 0.55
STABLE_FRAMES_NEEDED = 20
MAX_SENTENCE_LEN     = 60

COL_BG          = (18,  18,  35)
COL_ACCENT      = (0,   210, 150)
COL_ACCENT2     = (40,  130, 255)
COL_WARNING     = (20,  90,  230)
COL_TEXT        = (235, 235, 255)
COL_SHADOW      = (0,   0,   0)
COL_CARD        = (30,  30,  55)

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

def extract_and_normalise(hand_landmarks):
    lm = hand_landmarks.landmark
    xs = np.array([p.x for p in lm])
    ys = np.array([p.y for p in lm])
    zs = np.array([p.z for p in lm])

    xs -= xs[0]; ys -= ys[0]; zs -= zs[0]

    scale = np.sqrt((xs.max() - xs.min())**2 + (ys.max() - ys.min())**2) + 1e-6
    xs /= scale; ys /= scale; zs /= scale

    return np.concatenate([xs, ys, zs]).reshape(1, -1)

def rounded_rect(img, x1, y1, x2, y2, r, color, alpha=0.75):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def shadow_text(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                scale=1.0, color=COL_TEXT, thickness=2):
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), font, scale,
                COL_SHADOW, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale,
                color, thickness, cv2.LINE_AA)

def confidence_bar(img, x, y, w, h, conf, label, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 70), -1)
    fill = int(conf * w)
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 100), 1)
    pct_text = f"{conf*100:.1f}%"
    cv2.putText(img, pct_text, (x + fill + 5, y + h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1, cv2.LINE_AA)

def hand_bounding_box(frame, hand_landmarks):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    pad = 20
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(w, int(max(xs)) + pad)
    y2 = min(h, int(max(ys)) + pad)
    return x1, y1, x2, y2

def main():
    for f in [MODEL_FILE, LABEL_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"'{f}' not found. Run train_model.py first.")

    print("[Loading] Model …", end=" ", flush=True)
    pipeline = joblib.load(MODEL_FILE)
    le       = joblib.load(LABEL_FILE)
    classes  = le.classes_
    print(f"OK  ({len(classes)} classes: {list(classes)})")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    pred_buffer     = deque(maxlen=SMOOTHING_WINDOW)
    sentence        = []
    stable_label    = None
    stable_count    = 0

    prev_time = time.time()

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as hands:

        print("\n[●] Live prediction started.")
        print("    C = clear sentence   SPACE = add space   Q/ESC = quit\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Webcam read failed.")
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            curr_time = time.time()
            fps       = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            predicted_label = None
            confidence      = 0.0
            proba_top5      = []
            bbox            = None

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style())

                    bbox = hand_bounding_box(frame, hand_lm)

                    feat = extract_and_normalise(hand_lm)

                    proba = pipeline.predict_proba(feat)[0]
                    top_idx = np.argsort(proba)[::-1]

                    confidence      = proba[top_idx[0]]
                    predicted_label = classes[top_idx[0]]

                    proba_top5 = [(classes[i], proba[i]) for i in top_idx[:5]]

                    if confidence >= CONFIDENCE_THRESHOLD:
                        pred_buffer.append(top_idx[0])

                if pred_buffer:
                    most_common_idx = Counter(pred_buffer).most_common(1)[0][0]
                    smooth_label    = classes[most_common_idx]

                    if smooth_label == stable_label:
                        stable_count += 1
                    else:
                        stable_label = smooth_label
                        stable_count = 1

                    if stable_count == STABLE_FRAMES_NEEDED:
                        if len("".join(sentence)) < MAX_SENTENCE_LEN:
                            sentence.append(smooth_label)
                        stable_count = 0

            else:
                pred_buffer.clear()
                stable_label  = None
                stable_count  = 0

            rounded_rect(frame, 10, 8, w - 10, 70, 12, COL_CARD, 0.80)
            shadow_text(frame, f"FPS {fps:5.1f}", (25, 52),
                        scale=0.9, color=COL_ACCENT)
            shadow_text(frame, "Sign Language Recognition  v3.0",
                        (165, 52), scale=0.85, color=COL_TEXT)
            shadow_text(frame, f"Classes: {len(classes)}",
                        (w - 165, 52), scale=0.8, color=COL_ACCENT2)

            if bbox and predicted_label and confidence >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = bbox
                box_col = COL_ACCENT if confidence > 0.80 else COL_ACCENT2
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2)
                L = 18
                for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                        (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                    cv2.line(frame, (px, py), (px + dx*L, py), box_col, 3)
                    cv2.line(frame, (px, py), (px, py + dy*L), box_col, 3)

                chip_text = f" {predicted_label} "
                (tw, th), _ = cv2.getTextSize(chip_text,
                                              cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
                chip_x1 = x1
                chip_y1 = max(0, y1 - th - 18)
                rounded_rect(frame, chip_x1, chip_y1,
                             chip_x1 + tw + 10, chip_y1 + th + 10,
                             8, box_col, 0.9)
                shadow_text(frame, chip_text, (chip_x1 + 5, chip_y1 + th + 2),
                            scale=1.4, color=(10, 10, 20), thickness=3)

            panel_w  = 260
            panel_x1 = w - panel_w - 15
            panel_y1 = 85
            panel_y2 = 85 + 320

            rounded_rect(frame, panel_x1, panel_y1, w - 15, panel_y2,
                         14, COL_CARD, 0.82)

            shadow_text(frame, "PREDICTION", (panel_x1 + 14, panel_y1 + 30),
                        scale=0.65, color=COL_ACCENT, thickness=1)
            cv2.line(frame, (panel_x1 + 10, panel_y1 + 38),
                     (w - 25, panel_y1 + 38), (60, 60, 90), 1)

            if predicted_label and confidence >= CONFIDENCE_THRESHOLD:
                shadow_text(frame, predicted_label,
                            (panel_x1 + 80, panel_y1 + 120),
                            scale=4.0, color=COL_ACCENT, thickness=6)
                shadow_text(frame, "Confidence",
                            (panel_x1 + 14, panel_y1 + 145),
                            scale=0.55, color=COL_TEXT, thickness=1)
                confidence_bar(frame, panel_x1 + 14, panel_y1 + 155,
                               panel_w - 28, 16, confidence,
                               predicted_label, COL_ACCENT)

                shadow_text(frame, "Top predictions",
                            (panel_x1 + 14, panel_y1 + 195),
                            scale=0.5, color=(160, 160, 190), thickness=1)
                for rank, (cls, prob) in enumerate(proba_top5):
                    ty = panel_y1 + 215 + rank * 22
                    bar_col = COL_ACCENT if rank == 0 else (70, 70, 100)
                    confidence_bar(frame, panel_x1 + 55, ty,
                                   panel_w - 75, 14, prob, cls, bar_col)
                    shadow_text(frame, f"{cls}:", (panel_x1 + 14, ty + 12),
                                scale=0.5, color=COL_TEXT, thickness=1)
            else:
                no_msg = ("No hand" if not results.multi_hand_landmarks
                          else "Low conf.")
                shadow_text(frame, no_msg, (panel_x1 + 35, panel_y1 + 100),
                            scale=1.0, color=(120, 120, 160), thickness=2)

            sentence_str = "".join(sentence)
            sb_y1 = h - 110
            rounded_rect(frame, 10, sb_y1, w - panel_w - 30, h - 15,
                         12, COL_CARD, 0.80)
            shadow_text(frame, "Sentence", (25, sb_y1 + 25),
                        scale=0.6, color=COL_ACCENT, thickness=1)
            cv2.line(frame, (15, sb_y1 + 33), (w - panel_w - 35, sb_y1 + 33),
                     (60, 60, 90), 1)

            display_sent = sentence_str[-45:] if len(sentence_str) > 45 else sentence_str
            if stable_label and stable_count > 0:
                display_sent += f"[{stable_label}·{stable_count}]"

            shadow_text(frame, display_sent if display_sent else "—",
                        (25, sb_y1 + 72),
                        scale=1.0,
                        color=COL_TEXT if sentence_str else (90, 90, 120),
                        thickness=2)

            shadow_text(frame, "C=clear  SPACE=space  Q=quit",
                        (25, h - 20),
                        scale=0.5, color=(120, 120, 160), thickness=1)

            if stable_label and stable_count > 0:
                angle_end = int((stable_count / STABLE_FRAMES_NEEDED) * 360)
                centre = (panel_x1 + panel_w // 2, panel_y1 + 310)
                cv2.ellipse(frame, centre, (22, 22), -90, 0,
                            angle_end, COL_ACCENT, 3)
                shadow_text(frame, stable_label, (centre[0] - 9, centre[1] + 8),
                            scale=0.7, color=COL_ACCENT, thickness=2)

            cv2.imshow("Sign Language — Live Prediction", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c') or key == ord('C'):
                sentence.clear()
                stable_label  = None
                stable_count  = 0
                pred_buffer.clear()
                print("[C] Sentence cleared.")
            elif key == ord(' '):
                sentence.append(" ")
                print("[SPACE] Space added.")

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Prediction session ended.")


if __name__ == "__main__":
    main()
