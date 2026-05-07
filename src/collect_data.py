import os
import cv2
import mediapipe as mp
import numpy as np

# New MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks(frame, hand_landmarks):
    """Draw hand landmarks on frame"""
    h, w, _ = frame.shape
    points = []

    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]

    for s, e in connections:
        if s < len(points) and e < len(points):
            cv2.line(frame, points[s], points[e], (0, 255, 0), 2)

    return frame


def collect_data(data_dir, classes, num_samples=200):
    """Collect hand gesture images using webcam"""

    # Create directories
    for cls in classes:
        os.makedirs(os.path.join(data_dir, cls), exist_ok=True)
    print(f"📁 Created {len(classes)} directories")

    # Model path
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "hand_landmarker.task"
    )

    if not os.path.exists(model_path):
        print(f"❌ hand_landmarker.task not found at: {model_path}")
        return

    # Create landmarker with num_hands=2
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    landmarker = HandLandmarker.create_from_options(options)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        landmarker.close()
        return

    print("\n" + "=" * 55)
    print("🤟 ISL DATA COLLECTOR (Supports 1 & 2 hands)")
    print("=" * 55)
    print("📌 One-handed sign → show 1 hand")
    print("📌 Two-handed sign → show both hands")
    print("📌 Press 'S' to start | 'Q' to quit")
    print("=" * 55)

    for idx, class_name in enumerate(classes):
        print(f"\n[{idx+1}/{len(classes)}] 📸 Gesture: '{class_name}'")
        print(f"   Make the ISL sign for '{class_name}'")
        print(f"   Press 'S' to start collecting")

        # ── Wait for user to be ready ──────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # Detect hands for preview
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)

            # Draw ALL detected hands
            num_hands = 0
            if result.hand_landmarks:
                num_hands = len(result.hand_landmarks)
                for hand in result.hand_landmarks:
                    display = draw_landmarks(display, hand)

            # Top bar
            cv2.rectangle(display, (0, 0), (display.shape[1], 100),
                         (40, 40, 40), -1)

            # Letter and instruction
            cv2.putText(display,
                       f"[{idx+1}/{len(classes)}] Letter: '{class_name}'",
                       (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                       (255, 255, 0), 2)

            cv2.putText(display,
                       "Press 'S' to start collecting | 'Q' to quit",
                       (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (200, 200, 200), 1)

            # Hand detection status
            if num_hands > 0:
                status_color = (0, 255, 0)
                status_text = f"✓ Hands detected: {num_hands}"
            else:
                status_color = (0, 0, 255)
                status_text = "✗ No hand detected - show your hand!"

            cv2.putText(display, status_text,
                       (20, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       status_color, 2)

            cv2.imshow("ISL Data Collection", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                landmarker.close()
                print("\n👋 Collection stopped by user!")
                return

        # ── Collect samples ────────────────────────────────
        count = 0
        print(f"   ⏳ Collecting {num_samples} samples...")

        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # Detect hands
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)

            # Draw ALL detected hands
            num_hands = 0
            if result.hand_landmarks:
                num_hands = len(result.hand_landmarks)
                for hand in result.hand_landmarks:
                    display = draw_landmarks(display, hand)

            # Top bar
            cv2.rectangle(display, (0, 0), (display.shape[1], 100),
                         (40, 40, 40), -1)

            # Progress text
            progress = int((count / num_samples) * 100)
            cv2.putText(display,
                       f"'{class_name}': {count}/{num_samples} ({progress}%)",
                       (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                       (0, 255, 0), 2)

            # Hands detected
            cv2.putText(display,
                       f"Hands detected: {num_hands}",
                       (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 0), 2)

            # Progress bar
            bar_w = display.shape[1] - 40
            filled = int((count / num_samples) * bar_w)
            cv2.rectangle(display,
                         (20, 110), (20 + bar_w, 130),
                         (50, 50, 50), -1)
            cv2.rectangle(display,
                         (20, 110), (20 + filled, 130),
                         (0, 255, 0), -1)
            cv2.putText(display, f"{progress}%",
                       (20 + bar_w//2 - 15, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)

            # Bottom status
            if num_hands > 0:
                status = f"✓ {num_hands} hand(s) in frame - hold steady!"
                s_color = (0, 255, 0)
            else:
                status = "✗ No hand detected!"
                s_color = (0, 0, 255)

            cv2.putText(display, status,
                       (20, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       s_color, 2)

            cv2.imshow("ISL Data Collection", display)

            # Save original frame
            path = os.path.join(data_dir, class_name, f"{count}.jpg")
            cv2.imwrite(path, frame)
            count += 1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                landmarker.close()
                print("\n👋 Collection stopped by user!")
                return

        print(f"   ✅ Done! {num_samples} samples saved for '{class_name}'")

    # Done!
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    print("\n" + "=" * 55)
    print("🎉 DATA COLLECTION COMPLETE!")
    print("=" * 55)
    print(f"   Total classes: {len(classes)}")
    print(f"   Samples each:  {num_samples}")
    print(f"   Total images:  {len(classes) * num_samples}")
    print(f"   Saved in:      {os.path.abspath(data_dir)}")
    print("\nNext step:")
    print("   python create_dataset.py")
    print("=" * 55)


if __name__ == "__main__":
    PERSON_ID = input("Enter person ID (person_1 / person_2 / person_3): ").strip()
    DATA_DIR = os.path.join("..", "data", "raw", PERSON_ID)

    if PERSON_ID == "person_1":
        NUM_SAMPLES = 200   # Training (large)
    elif PERSON_ID == "person_2":
        NUM_SAMPLES = 50    # Validation
    elif PERSON_ID == "person_3":
        NUM_SAMPLES = 50    # Test
    else:
        print("❌ Invalid PERSON_ID! Use person_1 / person_2 / person_3")
        exit()
    
    # ALL 26 ISL ALPHABETS
    CLASSES = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z'
    ]

    print(f"\n📊 Collecting {NUM_SAMPLES} samples per class for {PERSON_ID}")

    collect_data(DATA_DIR, CLASSES, NUM_SAMPLES)