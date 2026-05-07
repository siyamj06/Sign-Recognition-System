import os
import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def extract_landmarks(data_dir, output_path):
    """Extract hand landmarks from collected images"""

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "hand_landmarker.task"
    )

    if not os.path.exists(model_path):
        print(f"❌ hand_landmarker.task not found!")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,                          # DETECT 2 HANDS
        min_hand_detection_confidence=0.5
    )
    landmarker = HandLandmarker.create_from_options(options)

    data = []
    labels = []
    skipped = 0

    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    print(f"📂 Found {len(classes)} classes: {classes}")
    print("=" * 50)

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        images = os.listdir(class_dir)
        print(f"\n🔄 Processing '{class_name}': {len(images)} images")

        class_count = 0

        for img_name in images:
            img_path = os.path.join(class_dir, img_name)

            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)

            if result.hand_landmarks and len(result.hand_landmarks) > 0:

                # Collect ALL landmarks from all detected hands
                all_x = []
                all_y = []

                for hand in result.hand_landmarks:
                    for lm in hand:
                        all_x.append(lm.x)
                        all_y.append(lm.y)

                # Normalize using global min
                min_x = min(all_x)
                min_y = min(all_y)

                feature_vector = []

                # Hand 1 landmarks (always present)
                hand1 = result.hand_landmarks[0]
                for lm in hand1:
                    feature_vector.append(lm.x - min_x)
                    feature_vector.append(lm.y - min_y)

                # Hand 2 landmarks (may or may not be present)
                if len(result.hand_landmarks) >= 2:
                    hand2 = result.hand_landmarks[1]
                    for lm in hand2:
                        feature_vector.append(lm.x - min_x)
                        feature_vector.append(lm.y - min_y)
                else:
                    # If only 1 hand detected, fill hand2 with zeros
                    feature_vector.extend([0.0] * 42)

                # Now feature_vector has 84 values
                # (21 landmarks × 2 coords × 2 hands)
                data.append(feature_vector)
                labels.append(class_name)
                class_count += 1
            else:
                skipped += 1

        print(f"   ✅ Extracted: {class_count} samples")

    landmarker.close()

    print("\n" + "=" * 50)
    print(f"📊 SUMMARY")
    print(f"   Extracted: {len(data)}")
    print(f"   Skipped: {skipped}")
    print(f"   Features: {len(data[0]) if data else 0}")
    print(f"   (21 landmarks × 2 coords × 2 hands = 84)")
    print("=" * 50)

    if not data:
        print("❌ No data extracted!")
        return

    # Save pickle
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"\n💾 Saved: {output_path}")

    # Save CSV
    csv_path = output_path.replace('.pkl', '.csv')
    columns = []
    for hand_num in range(1, 3):    # hand1, hand2
        for i in range(21):
            columns.extend([f'h{hand_num}_x_{i}', f'h{hand_num}_y_{i}'])

    df = pd.DataFrame(data, columns=columns)
    df['label'] = labels
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved: {csv_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.join("..", "data", "raw")

    PERSON_1_DIR = os.path.join(BASE_DIR, "person_1")
    PERSON_2_DIR = os.path.join(BASE_DIR, "person_2")
    PERSON_3_DIR = os.path.join(BASE_DIR, "person_3")

    PROCESSED_DIR = os.path.join("..", "data", "processed")

    print("\n🔹 Creating TRAIN dataset (person_1)")
    extract_landmarks(
        PERSON_1_DIR,
        os.path.join(PROCESSED_DIR, "train.pkl")
    )

    print("\n🔹 Creating TEST dataset (person_2)")
    extract_landmarks(
        PERSON_2_DIR,
        os.path.join(PROCESSED_DIR, "test.pkl")
    )

    print("\n🔹 Creating VALIDATION dataset (person_3)")
    extract_landmarks(
        PERSON_3_DIR,
        os.path.join(PROCESSED_DIR, "val.pkl")
    )

    print("\n🎉 ALL DATASETS CREATED SUCCESSFULLY!")