import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import time
from collections import deque

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class Recognizer:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            d = pickle.load(f)

        self.model = d['model']
        self.scaler = d.get('scaler', None)  # ✅ Load scaler
        self.model_name = d.get('model_name', 'Unknown Model')
        self.accuracy = d.get('accuracy', 0.0)
        
        self.labels = d.get('labels', None)
        self.label_encoder = d.get('label_encoder', None)
        
        if self.labels:
            print(f"✅ {self.model_name} | Accuracy: {self.accuracy*100:.2f}%")
            print(f"   Classes: {self.labels}")
        else:
            print(f"✅ {self.model_name} | Accuracy: {self.accuracy*100:.2f}%")
            print(f"   ⚠️ No label list found in model file")
            
            if self.label_encoder:
                self.labels = list(self.label_encoder.classes_)
                print(f"   📋 Inferred classes: {self.labels}")

        lm_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "hand_landmarker.task"
        )

        if not os.path.exists(lm_path):
            print(f"❌ hand_landmarker.task not found!")
            raise FileNotFoundError("hand_landmarker.task missing")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=lm_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        print("✅ Hand Landmarker ready (1 & 2 hands)")
        
        if self.scaler:
            print("✅ Scaler loaded")
        else:
            print("⚠️ No scaler found - predictions may be inaccurate!")


        self.buffer = deque(maxlen=8)
        self.sentence = []
        self.last_letter = None
        self.last_time = 0

    def extract(self, hands_list):
        """Extract features from 1 or 2 hands (84 features)"""

        all_x = []
        all_y = []
        for hand in hands_list:
            for lm in hand:
                all_x.append(lm.x)
                all_y.append(lm.y)

        min_x = min(all_x)
        min_y = min(all_y)

        features = []

        # Hand 1 → 42 features
        hand1 = hands_list[0]
        for lm in hand1:
            features.append(lm.x - min_x)
            features.append(lm.y - min_y)

        # Hand 2 → 42 features (or zeros)
        if len(hands_list) >= 2:
            hand2 = hands_list[1]
            for lm in hand2:
                features.append(lm.x - min_x)
                features.append(lm.y - min_y)
        else:
            features.extend([0.0] * 42)

        return features

    def mirror_features(self, features):
        """Mirror x-coordinates per hand block to normalize left/right orientation."""
        mirrored = np.array(features, dtype=np.float32).copy()

        # Each hand contributes 42 values: (x0, y0, ..., x20, y20)
        for hand_start in (0, 42):
            hand_block = mirrored[hand_start:hand_start + 42]

            # Skip zero-padded hand block when only one hand is present.
            if np.allclose(hand_block, 0.0):
                continue

            x_idx = np.arange(0, 42, 2)
            max_x = np.max(hand_block[x_idx])
            hand_block[x_idx] = max_x - hand_block[x_idx]
            mirrored[hand_start:hand_start + 42] = hand_block

        return mirrored

    def smooth(self, pred):
        self.buffer.append(pred)
        if len(self.buffer) < 5:
            return pred
        return max(set(self.buffer), key=list(self.buffer).count)

    def draw_hand(self, frame, hand):
        h, w, _ = frame.shape
        pts = []
        for lm in hand:
            x, y = int(lm.x * w), int(lm.y * h)
            pts.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        conns = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
        ]
        for s, e in conns:
            if s < len(pts) and e < len(pts):
                cv2.line(frame, pts[s], pts[e], (0, 255, 0), 2)
        return frame, pts

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return

        print("\n" + "=" * 55)
        print("🤟 ISL SIGN LANGUAGE RECOGNITION")
        print("=" * 55)
        print("Supports ONE and TWO handed signs!")
        print()
        print("Controls:")
        print("  SPACE     → Add space")
        print("  BACKSPACE → Delete last")
        print("  C         → Clear sentence")
        print("  Q         → Quit")
        print("=" * 55)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_img)

            prediction = None
            confidence = 0
            num_hands = 0

            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                num_hands = len(result.hand_landmarks)

                all_pts = []
                for hand in result.hand_landmarks:
                    frame, pts = self.draw_hand(frame, hand)
                    all_pts.extend(pts)

                if all_pts:
                    xs = [p[0] for p in all_pts]
                    ys = [p[1] for p in all_pts]
                    pad = 20
                    x1 = max(0, min(xs) - pad)
                    y1 = max(0, min(ys) - pad)
                    x2 = min(w, max(xs) + pad)
                    y2 = min(h, max(ys) + pad)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                features = self.extract(result.hand_landmarks)
                feat_original = np.array(features, dtype=np.float32).reshape(1, -1)
                feat_mirrored = self.mirror_features(features).reshape(1, -1)

                # ✅ SCALE FEATURES
                if self.scaler is not None:
                    feat_original = self.scaler.transform(feat_original)
                    feat_mirrored = self.scaler.transform(feat_mirrored)

                if hasattr(self.model, 'predict_proba'):
                    proba_original = self.model.predict_proba(feat_original)
                    proba_mirrored = self.model.predict_proba(feat_mirrored)

                    if np.max(proba_mirrored) > np.max(proba_original):
                        proba = proba_mirrored
                    else:
                        proba = proba_original

                    class_idx = int(np.argmax(proba, axis=1)[0])
                    raw = self.model.classes_[class_idx]
                    confidence = float(np.max(proba))
                else:
                    raw = self.model.predict(feat_original)[0]
                    confidence = 0.9
                
                if self.label_encoder is not None:
                    prediction_decoded = self.label_encoder.inverse_transform([raw])[0]
                    prediction = self.smooth(prediction_decoded)
                else:
                    prediction = self.smooth(raw)

                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                cv2.putText(frame, f"{prediction} ({confidence*100:.0f}%)",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           1.2, color, 3)

                now = time.time()
                if prediction == self.last_letter:
                    if now - self.last_time > 1.2:
                        self.sentence.append(prediction)
                        self.last_time = now
                        print(f"   ✅ Added: {prediction}")
                else:
                    self.last_letter = prediction
                    self.last_time = now
            else:
                self.buffer.clear()

            cv2.rectangle(frame, (0, 0), (w, 90), (40, 40, 40), -1)
            cv2.putText(frame, "ISL Sign Language Recognition",
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 255), 2)

            if prediction:
                cv2.putText(frame,
                           f"Detected: {prediction} | Confidence: {confidence*100:.0f}%",
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame,
                           "Show your hand gesture...",
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (100, 100, 255), 2)

            cv2.putText(frame, f"Hands: {num_hands}",
                       (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 0), 2)

            cv2.rectangle(frame, (0, h - 80), (w, h), (40, 40, 40), -1)
            text = "".join(self.sentence)
            cv2.putText(frame, f"Sentence: {text}",
                       (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 0), 2)

            cv2.putText(frame,
                       "SPACE:Space | BACKSPACE:Delete | C:Clear | Q:Quit",
                       (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (200, 200, 200), 1)

            cv2.imshow("ISL Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            # Exit when user closes the OpenCV window from the title bar.
            try:
                if cv2.getWindowProperty("ISL Recognition", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.sentence.append(" ")
                print("   Added: [SPACE]")
            elif key == 8 and self.sentence:
                removed = self.sentence.pop()
                print(f"   Removed: {removed}")
            elif key == 13:
                t = "".join(self.sentence)
                if t.strip():
                    print(f"\n✅ RECOGNIZED: {t}\n")
                    self.sentence.clear()
            elif key == ord('c'):
                self.sentence.clear()
                print("   Cleared!")

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()
        print("\n👋 Stopped!")


if __name__ == "__main__":
    MODEL = os.path.join("..", "models", "sign_language_model.pkl")
    if not os.path.exists(MODEL):
        print("❌ Model not found!")
        print("   Run these first:")
        print("   1. python collect_data.py")
        print("   2. python create_dataset.py")
        print("   3. python train_model.py")
    else:
        Recognizer(MODEL).run()