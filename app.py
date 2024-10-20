import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from tensorflow.keras.models import load_model
from scipy import stats

# Actions that we try to detect
actions = np.array(['hello', 'how', 'you','fine'])

# Load the pre-trained model
model = load_model('sign.h5')

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

def main():
    # Custom CSS for background image and layout styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #FF6347; /* Dark background */
            color: white;  /* White text for contrast */
        }
        .header {
            font-family: 'Arial Black', sans-serif;
            font-size: 40px;
            color:#20242B ; /* Tomato color for the title */
            text-align: left;
            padding: 20px;
        }
        .caption {
            font-size: 20px;
            color: #E0E0E0;
        }
        .sidebar {
            background-color: #000000; /* Dark sidebar */
            color: white;
            padding: 20px;
        }
        .sidebar h2 {
            color: #FF6347; /* Tomato color for sidebar headers */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title and logo 
    
    st.markdown(
    """
    <style>
    .header {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        margin-bottom:5px;
    }
    .caption {
        text-align: center;
        font-size: 24px;
        color: white;
        font-style: italic;
        margin-top:0;
    }
    </style>
    <div class="header">SignPAL</div>
    <div class="caption">Bridging Conversations</div>
    """, 
    unsafe_allow_html=True
    )


    # Sidebar details with some styling
    with st.sidebar:
        st.image("logo.jpg", width=200)

        st.header("Details")
        
        # Objective Section
        with st.expander("Objective", expanded=False):
            st.write("""
            This application aims to bridge communication gaps for individuals who are non-verbal. 
            By translating sign language gestures into real-time closed captions in English, we strive to foster inclusivity and 
            create a more connected community where everyone can communicate effectively.
            """)

        # How to Use Instructions
        with st.expander("How to Use", expanded=False):
            st.write("""
            1. *Open the Application*: Launch the app in your web browser.
            2. *Start the Camera*: Click on the 'Start' button to enable your camera.
            3. *Record Sign Language*: When your friend begins to sign, click on the 'Record' button.
            4. *Real-Time Translation*: The app will convert the sign language into English captions in real-time.
            5. *Stop Recording*: Click on the 'Stop' button when you are finished.
            """)

        # About the Team Section
        with st.expander("About the Team", expanded=False):
            st.write("""
            We are *Pumpkin Seeds*, a group of passionate individuals dedicated to creating solutions that enhance communication for everyone. 
            Our team comprises graduate students from Clark University pursuing a Master's in Data Analytics:
            - Kunal Malhan
            - Keerthana Goka
            - Jothsna Praveena Pendyala
            - Mohan Manem
            """)



    
    st.write("The model will process video input and translate sign language gestures in real-time.")
    

    video_placeholder = st.empty()
    stop_button = st.button("Stop")

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Failed to open webcam. Please check your camera connection.")
        return

    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("Failed to capture frame. Retrying...")
                continue

            image, results = mediapipe_detection(frame, holistic)
            # draw_styled_landmarks(image, results)

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

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (4,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            video_placeholder.image(image, channels="BGR")

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    st.write("Webcam released. You can close the app now.")

if __name__ == '__main__':
    main()