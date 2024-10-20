import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from tensorflow.keras.models import load_model
from scipy import stats

# Actions that we try to detect
actions = np.array(['hello', 'how', 'you', 'fine'])

# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Initialize session state for camera control
if 'is_camera_on' not in st.session_state:
    st.session_state.is_camera_on = False

def main():
    # Load model and initialize MediaPipe
    try:
        model = load_model('sign.h5')
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Your existing CSS styles
    st.markdown("""
        
        """, unsafe_allow_html=True)
    # Title and Header
    st.markdown("""
        

SignPAL


        

Bridging Conversations


        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("logo.jpg", width=200)
        st.header("Details")
        
        with st.expander("Objective", expanded=False):
            st.write("""
            This application aims to bridge communication gaps for individuals who are non-verbal. 
            By translating sign language gestures into real-time closed captions in English, we strive to foster inclusivity and 
            create a more connected community where everyone can communicate effectively.
            """)

        with st.expander("How to Use", expanded=False):
            st.write("""
            1. Click 'Start Camera' to enable your camera
            2. Perform sign language gestures
            3. The app will translate gestures in real-time
            4. Click 'Stop Camera' when finished
            """)

        with st.expander("About the Team", expanded=False):
            st.write("""
            We are *Pumpkin Seeds*, a team of Clark University graduate students:
            - Kunal Malhan
            - Keerthana Goka
            - Jothsna Praveena Pendyala
            - Mohan Manem
            """)

    # Main content
    st.write("The model will process video input and translate sign language gestures in real-time.")

    # Camera controls
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start Camera", disabled=st.session_state.is_camera_on)
    with col2:
        stop_button = st.button("Stop Camera", disabled=not st.session_state.is_camera_on)

    # Video placeholder
    video_placeholder = st.empty()

    # Camera handling
    if start_button:
        st.session_state.is_camera_on = True
        
    if stop_button:
        st.session_state.is_camera_on = False
    if st.session_state.is_camera_on:
        try:
            # Use Streamlit's built-in camera access
            video_capture = st.camera_input("Capture Video")

            if video_capture:
                # Read the video stream
                cap = cv2.VideoCapture(video_capture)
                if not cap.isOpened():
                    st.error("Failed to open webcam. Please check your camera connection.")
                    st.session_state.is_camera_on = False
                    return

                sequence = []
                sentence = []
                predictions = []
                threshold = 0.7

                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    while cap.isOpened() and st.session_state.is_camera_on:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("Failed to capture frame. Retrying...")
                            continue

                        # Process frame
                        image, results = mediapipe_detection(frame, holistic)
                        keypoints = extract_keypoints(results)
                        sequence.append(keypoints)
                        sequence = sequence[-30:]

                        if len(sequence) == 30:
                            res = model.predict(np.expand_dims(sequence, axis=0))[0]
                            predictions.append(np.argmax(res))

                            if np.unique(predictions[-10:])[0] == np.argmax(res):
                                if res[np.argmax(res)] > threshold:
                                    if len(sentence) > 0:
                                        if actions[np.argmax(res)] != sentence[-1]:
                                            sentence.append(actions[np.argmax(res)])
                                    else:
                                        sentence.append(actions[np.argmax(res)])

                            if len(sentence) > 5:
                                sentence = sentence[-5:]

                        # Draw text on frame
                        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                        cv2.putText(image, ' '.join(sentence), (4,30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # Display frame
                        video_placeholder.image(image, channels="BGR")

                cap.release()
                cv2.destroyAllWindows()
                
        except Exception as e:
            st.error(f"Error during camera operation: {str(e)}")
            st.session_state.is_camera_on = False

if __name__ == '__main__':
    main()
