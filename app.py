import streamlit as st
import cv2
import pickle
import numpy as np
import mediapipe as mp
import os
import time

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Page config
st.set_page_config(
    page_title="ISL Recognition System",
    page_icon="🤟",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = os.path.join("models", "sign_language_model.pkl")
    
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_landmarker():
    """Load the hand landmarker"""
    model_path = "hand_landmarker.task"
    
    if not os.path.exists(model_path):
        return None
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,  
        min_hand_detection_confidence=0.5
    )
    try:
        return HandLandmarker.create_from_options(options)
    except OSError as exc:
        st.error("❌ MediaPipe hand landmarker could not load on this environment.")
        st.caption(f"Native library error: {exc}")
        return None


def extract_landmarks(hands_list):
    """Extract normalized features from 1 or 2 hands"""
    if not hands_list:
        return None
    
    # Get all coordinates for normalization
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
    for lm in hands_list[0]:
        features.append(lm.x - min_x)
        features.append(lm.y - min_y)
    
    # Hand 2 → 42 features (or zeros if only 1 hand)
    if len(hands_list) >= 2:
        for lm in hands_list[1]:
            features.append(lm.x - min_x)
            features.append(lm.y - min_y)
    else:
        features.extend([0.0] * 42)
    
    return features


def draw_hands(frame_rgb, hands_list):
    """Draw landmarks on all detected hands"""
    h, w, _ = frame_rgb.shape

    for hand in hands_list:
        points = []
        
        # Draw landmarks
        for lm in hand:
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame_rgb, (x, y), 5, (0, 255, 0), -1)

        # Draw connections
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
                cv2.line(frame_rgb, points[s], points[e], (0, 255, 0), 2)

    return frame_rgb
def predict_gesture(features, model_data):
    """Predict gesture using original + mirrored landmarks to support both hands."""

    def mirror_feature_vector(feature_vec):
        mirrored = np.array(feature_vec, dtype=np.float32).copy()

        # Each hand contributes 42 values: (x0, y0, ..., x20, y20)
        for hand_start in (0, 42):
            hand_block = mirrored[hand_start:hand_start + 42]

            # Zero-padded hand (no detected hand)
            if np.allclose(hand_block, 0.0):
                continue

            x_idx = np.arange(0, 42, 2)
            max_x = np.max(hand_block[x_idx])
            hand_block[x_idx] = max_x - hand_block[x_idx]
            mirrored[hand_start:hand_start + 42] = hand_block

        return mirrored

    model = model_data['model']
    label_encoder = model_data.get('label_encoder', None)
    scaler = model_data.get('scaler', None)

    original = np.array(features, dtype=np.float32).reshape(1, -1)
    mirrored = mirror_feature_vector(features).reshape(1, -1)

    if scaler is not None:
        original = scaler.transform(original)
        mirrored = scaler.transform(mirrored)

    # Prefer probability-based selection when available.
    if hasattr(model, 'predict_proba'):
        proba_original = model.predict_proba(original)
        proba_mirrored = model.predict_proba(mirrored)

        if np.max(proba_mirrored) > np.max(proba_original):
            proba = proba_mirrored
        else:
            proba = proba_original

        raw_prediction = np.argmax(proba, axis=1)[0]
        confidence = float(np.max(proba))
    else:
        raw_prediction = model.predict(original)[0]
        confidence = 0.9
        proba = None

    if label_encoder is not None:
        prediction = label_encoder.inverse_transform([raw_prediction])[0]
    else:
        prediction = raw_prediction

    return prediction, confidence, proba


def main():
    # ═══════════════════════════════════════════════
    # CUSTOM STYLING - BEIGE & BROWN THEME
    # ═══════════════════════════════════════════════
    st.markdown("""
    <style>
        /* ══════════════════════════════════════════ */
        /* BACKGROUNDS */
        /* ══════════════════════════════════════════ */
        .main {
            background-color: #F5F1E8;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #F5F1E8;
        }
        [data-testid="stSidebar"] {
            background-color: #FFFAF0;
        }
        
        /* ══════════════════════════════════════════ */
        /* TEXT COLORS */
        /* ══════════════════════════════════════════ */
        body, .main, [class*="st-"] {
            color: #333333;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #A67B5B !important;
        }
        
        p, span, li {
            color: #333333;
        }
        
        /* ══════════════════════════════════════════ */
        /* BUTTONS */
        /* ══════════════════════════════════════════ */
        .stButton > button {
            background-color: #A67B5B;
            color: white;
            border: 2px solid #A67B5B;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #8B5E3C;
            border-color: #8B5E3C;
            box-shadow: 0 4px 8px rgba(139, 94, 60, 0.3);
        }
        
        .stButton > button:active {
            background-color: #744A2E;
            border-color: #744A2E;
        }
        
        /* ══════════════════════════════════════════ */
        /* FILE UPLOADER */
        /* ══════════════════════════════════════════ */
        [data-testid="stFileUploadDropzone"] {
            background-color: #FFFAF0 !important;
            border: 2px dashed #A67B5B !important;
            border-radius: 8px;
        }
        
        [data-testid="stFileUploadDropzone"] > div {
            background-color: #FFFAF0 !important;
        }
        
        .stFileUploader section {
            background-color: #FFFAF0 !important;
            border: 2px dashed #A67B5B !important;
        }
        
        .stFileUploader [data-testid="stFileUploadDropzone"] {
            background-color: #FFFAF0 !important;
            border: 2px dashed #A67B5B !important;
        }
        
        /* File upload drag area */
        div[data-testid="stFileUploadDropzone"] {
            background-color: #FFFAF0 !important;
            border: 2px dashed #A67B5B !important;
            padding: 30px;
        }
        
        /* Upload button */
        .stFileUploader button {
            background-color: #A67B5B !important;
            color: white !important;
            border: 1px solid #A67B5B !important;
        }
        
        .uploadedFile {
            background-color: #F5F1E8 !important;
            border-left: 4px solid #A67B5B !important;
        }
        
        /* ══════════════════════════════════════════ */
        /* TABS */
        /* ══════════════════════════════════════════ */
        [data-baseweb="tab-list"] {
            background-color: transparent;
        }
        
        [data-baseweb="tab"] {
            color: #333333;
            border-bottom: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        [data-baseweb="tab"]:hover {
            color: #A67B5B;
            border-bottom-color: #D4A574;
        }
        
        [aria-selected="true"] {
            color: #A67B5B !important;
            border-bottom-color: #A67B5B !important;
        }
        
        /* ══════════════════════════════════════════ */
        /* SIDEBAR ELEMENTS */
        /* ══════════════════════════════════════════ */
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background-color: #F5F1E8;
            border: 1px solid #D4A574;
        }
        
        [data-testid="stSidebar"] [data-baseweb="radio"] {
            color: #333333;
        }
        
        /* ══════════════════════════════════════════ */
        /* METRICS */
        /* ══════════════════════════════════════════ */
        [data-testid="metric-container"] {
            background-color: #FFFAF0;
            border: 2px solid #D4A574;
            border-radius: 8px;
            padding: 15px;
        }
        
        [data-testid="metric-container"] > div > span {
            color: #333333;
        }
        
        /* ══════════════════════════════════════════ */
        /* DIVIDERS & BORDERS */
        /* ══════════════════════════════════════════ */
        hr {
            background-color: #D4A574;
            height: 1px;
            border: none;
        }
        
        /* ══════════════════════════════════════════ */
        /* EXPANDERS */
        /* ══════════════════════════════════════════ */
        [data-testid="stExpander"] {
            border: 1px solid #D4A574;
            border-radius: 6px;
        }
        
        [data-testid="stExpander"] > div > div > button {
            color: #A67B5B;
        }
        
        /* ══════════════════════════════════════════ */
        /* INPUTS */
        /* ══════════════════════════════════════════ */
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] {
            background-color: #FFFAF0;
            border: 1px solid #D4A574 !important;
            color: #333333;
        }
        
        [data-baseweb="input"] input:focus,
        [data-baseweb="textarea"] textarea:focus {
            border-color: #A67B5B !important;
            box-shadow: 0 0 0 2px rgba(166, 123, 91, 0.1);
        }
        
        /* ══════════════════════════════════════════ */
        /* ALERTS & INFO BOXES */
        /* ══════════════════════════════════════════ */
        [data-testid="stAlert"] {
            background-color: #FFFAF0;
            border-left: 4px solid #A67B5B;
        }
        
        /* SVG icon colors in uploader */
        svg {
            color: #A67B5B !important;
        }
        
        svg path, svg circle, svg line {
            stroke: #A67B5B !important;
            fill: #A67B5B !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════
    st.title("🤟 Indian Sign Language Recognition System")
    st.markdown("""
    **Real-time ISL alphabet recognition using Machine Learning & Computer Vision**
    
    *B.Tech AI & ML - 2nd Year Final Project*
    """)
    st.divider()

    # ═══════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════
    with st.sidebar:
        st.header("📊 Model Information")
        
        model_data = load_model()
        
        if model_data is None:
            st.error("❌ Model not found!")
            st.info("Train the model first:\n```bash\npython src/train_model.py\n```")
            return
        
        st.success("✅ Model Loaded")
        st.metric("Model Type", model_data['model_name'])
        st.metric("Test Accuracy", f"{model_data['accuracy'] * 100:.2f}%")
        
        # Get labels from label_encoder if not in model_data
        labels = model_data.get('labels', None)
        if labels is None and model_data.get('label_encoder'):
            labels = list(model_data['label_encoder'].classes_)
        
        if labels:
            st.write(f"**Total Classes:** {len(labels)}")
            with st.expander("📋 View All Classes"):
                st.write(", ".join(labels))
        
        st.divider()
        
        st.header("ℹ️ About")
        st.info("""
        **Technology Stack:**
        - MediaPipe (Hand Detection)
        - Scikit-learn (ML Models)
        - OpenCV (Image Processing)
        - Streamlit (Web Interface)
        
        **Supports:**
        - One-handed signs ✋
        - Two-handed signs 🙌
        - 26 ISL alphabet letters
        """)

    # ═══════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload Image",
        "📸 Webcam Capture",
        "📚 ISL Reference",
        "📊 Model Performance"
    ])

    # ═══════════════════════════════════════════════
    # TAB 1: Upload Image
    # ═══════════════════════════════════════════════
    with tab1:
        st.header("📤 Upload Hand Gesture Image")
        
        uploaded = st.file_uploader(
            "Choose an image showing ISL hand gesture",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo with good lighting"
        )

        if uploaded:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Original Image")
                st.image(image_rgb, width="stretch")

            with col2:
                st.subheader("🔍 Recognition Result")

                landmarker = load_landmarker()
                
                if landmarker is None:
                    st.error("❌ hand_landmarker.task not found!")
                    st.info("Download it from MediaPipe models repository")
                    return

                # Detect hands
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=image_rgb
                )
                result = landmarker.detect(mp_image)

                if result.hand_landmarks and len(result.hand_landmarks) > 0:
                    # Draw all hands
                    annotated = image_rgb.copy()
                    annotated = draw_hands(annotated, result.hand_landmarks)

                    st.image(annotated, width="stretch")

                    # Extract features (handles 1 or 2 hands)
                    features = extract_landmarks(result.hand_landmarks)
                    
                    # ✅ Predict with scaler
                    prediction, confidence, proba = predict_gesture(features, model_data)

                    # Display result
                    st.success(f"### Predicted: **{prediction}**")
                    st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
                    
                    st.info(f"**Hands Detected:** {len(result.hand_landmarks)}")

                    # Top 3 predictions
                    if proba is not None:
                        st.subheader("Top 3 Predictions:")
                        proba_flat = proba.flatten()
                        top_idx = np.argsort(proba_flat)[::-1][:3]
                        
                        labels = model_data.get('labels', None)
                        if labels is None:
                            labels = list(model_data['label_encoder'].classes_)
                        
                        for idx in top_idx:
                            label = labels[idx]
                            prob = proba_flat[idx]
                            st.write(f"**{label}**: {prob*100:.2f}%")
                else:
                    st.error("❌ No hand detected!")
                    st.info("💡 Tips:\n- Use good lighting\n- Show hand clearly\n- Use plain background")

    # ═══════════════════════════════════════════════
    # TAB 2: Webcam Capture
    # ═══════════════════════════════════════════════
    with tab2:
        st.header("📹 Live Webcam Recognition")
        st.info("💡 **Best Experience:** Use the desktop app for full live recognition")

        if 'webcam_enabled' not in st.session_state:
            st.session_state.webcam_enabled = False
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.code("cd src\npython recognise.py", language="bash")
            st.write("**Desktop app features:**")
            st.write("✅ Continuous live recognition")
            st.write("✅ Sentence building")
            st.write("✅ Text-to-speech")
            st.write("✅ Smooth predictions")
            st.write("✅ 25-30 FPS performance")
        
        with col2:
            st.write("**Web app (Single Capture):**")
            st.write("⚡ Quick test recognition")
            st.write("📸 Snapshot-based")
            st.write("🌐 Access from browser")
        
        st.markdown("---")
        
        st.subheader("📸 Quick Snapshot Recognition")
        st.caption("Works in browser and Streamlit Cloud using your device camera.")
        control_col1, control_col2 = st.columns(2)
        with control_col1:
            if st.button("📷 Capture Image", width="stretch"):
                st.session_state.webcam_enabled = True
        with control_col2:
            if st.button("⏹️ Reset", width="stretch"):
                st.session_state.webcam_enabled = False

        if not st.session_state.webcam_enabled:
            st.info("Open this tab and click **Capture Image** to enable capture.")
            captured = None
        else:
            captured = st.camera_input("Take a snapshot", key="tab2_camera")
            if captured is None:
                st.info("Allow browser camera access, then take a snapshot.")

        if captured is not None:
            landmarker = load_landmarker()

            if landmarker is None:
                st.error("❌ hand_landmarker.task not found!")
                return

            file_bytes = np.asarray(bytearray(captured.getvalue()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is None:
                st.error("❌ Failed to read captured image.")
                return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.flip(frame_rgb, 1)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Captured Frame")
                st.image(frame_rgb, width="stretch")

            with col2:
                st.subheader("🔍 Result")

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb
                )
                result = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    annotated = frame_rgb.copy()
                    annotated = draw_hands(annotated, result.hand_landmarks)
                    st.image(annotated, width="stretch")

                    features = extract_landmarks(result.hand_landmarks)
                    prediction, confidence, _ = predict_gesture(features, model_data)

                    st.success(f"### Predicted: **{prediction}**")
                    st.progress(confidence, text=f"{confidence*100:.1f}%")
                else:
                    st.error("No hand detected in captured image!")

    # ═══════════════════════════════════════════════
    # TAB 3: ISL Reference
    # ═══════════════════════════════════════════════
    with tab3:
        st.header("📚 Indian Sign Language Alphabet Reference")
        
        st.image(
            "https://static.vikaspedia.in/mediastorage/image/aphabets_and_numbers_in_isl.png",
            caption="Indian Sign Language Character Chart",
            width="stretch"
        )
        
        st.markdown("""
        ### 📖 How to Use This System
        
        **Upload Image Tab:**
        - Upload a clear photo of your hand gesture
        - System detects hand and recognizes the letter
        - Shows confidence score and top predictions
        
        **Webcam Capture Tab:**
        - Use browser camera to capture a snapshot
        - System recognizes the gesture instantly
        
        **Desktop App (Recommended):**
        - For continuous real-time recognition
        - Run: `python src/recognise.py`
        - Build sentences letter by letter
        
        ### ✅ Supported Features
        - ✋ One-handed ISL signs
        - 🙌 Two-handed ISL signs
        - 🔤 26 alphabet letters (A-Z)
        - 📊 Confidence scores
        - 🎯 High accuracy predictions
        
        ### 💡 Tips for Best Results
        - 🌟 **Lighting:** Face a window or use good lighting
        - 🌟 **Background:** Use plain white wall
        - 🌟 **Hand Position:** Keep hand clearly visible
        - 🌟 **Gesture:** Form the sign properly
        - 🌟 **Distance:** Keep hand 1-2 feet from camera
        """)

    # ═══════════════════════════════════════════════
    # TAB 4: Model Performance
    # ═══════════════════════════════════════════════
    with tab4:
        st.header("📊 Model Performance Analysis")
        
        plots_dir = "plots"
        
        if os.path.exists(plots_dir):
            # Model Comparison
            st.subheader("🔬 Model Comparison")
            comp_path = os.path.join(plots_dir, "model_comparison.png")
            if os.path.exists(comp_path):
                st.image(comp_path, width="stretch")
                st.caption("Comparison of different ML algorithms on test data")
            
            st.divider()
            
            # Confusion Matrix
            st.subheader("🎯 Confusion Matrix")
            cm_path = os.path.join(plots_dir, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, width="stretch")
                st.caption("Shows which letters are sometimes confused by the model")
            
            st.divider()
            
            # Class Distribution
            st.subheader("📈 Dataset Distribution")
            cd_path = os.path.join(plots_dir, "class_distribution.png")
            if os.path.exists(cd_path):
                st.image(cd_path, width="stretch")
                st.caption("Number of training samples per gesture class")
        else:
            st.warning("⚠️ Plots not found. Train the model to generate performance charts.")
            st.code("python src/train_model.py", language="bash")
        
        st.divider()
        
        # Model Metrics
        st.subheader("📈 Model Metrics")
        
        labels = model_data.get('labels', None)
        if labels is None and model_data.get('label_encoder'):
            labels = list(model_data['label_encoder'].classes_)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", model_data['model_name'])
        
        with col2:
            st.metric("Test Accuracy", f"{model_data['accuracy']*100:.2f}%")
        
        with col3:
            if labels:
                st.metric("Total Classes", len(labels))
        
        # Technical Details
        st.divider()
        st.subheader("🔧 Technical Details")
        
        num_classes = len(labels) if labels else "N/A"
        
        st.markdown(f"""
        **Model:** {model_data['model_name']}
        
        **Features:**
        - 84 features per sample (21 landmarks × 2 coordinates × 2 hands)
        - Coordinates normalized relative to hand position (min_x, min_y)
        - Position-invariant representation
        - Handles both single-hand and dual-hand gestures via zero-padding
        - StandardScaler normalization applied (zero mean, unit variance)
        
        **Training:**
        - Training Data: person_1 dataset
        - Validation Data: person_3 dataset
        - Test Data: person_2 dataset
        - Cross-Validation: 5-fold (on training data only)
        - No random train/test split (person-wise split for better generalization)
        
        **Performance:**
        - Accuracy: {model_data['accuracy']*100:.2f}%
        - Classes: {num_classes} ISL alphabet letters
        - Real-time FPS: ~25-30 frames per second
        """)


if __name__ == "__main__":
    main()