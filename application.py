import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
import av
import streamlit_webrtc as webrtc

# Load the pre-trained model
model = load_model(r'my_model3.h5')

def detect_mask(frame):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey:ey+eh, ex:ex+ew]  # cropping the face
                final_image = cv2.resize(face_roi, (224, 224))
                final_image = np.expand_dims(final_image, axis=0)
                final_image = final_image / 255.0
                font = cv2.FONT_HERSHEY_SIMPLEX
                predictions = model.predict(final_image)

                font_scale = 1.5
                font = cv2.FONT_HERSHEY_PLAIN

                if predictions > 0:
                    status = "No Mask"
                    x1, y1, w1, h1 = 0, 0, 175, 75
                    # Draw black background rectangle
                    cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 0, 0), -1)

                    # Add text
                    cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
                else:
                    status = "Face Mask"
                    x1, y1, w1, h1 = 0, 0, 175, 75
                    # Draw black background rectangle
                    cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 0, 0), -1)

                    # Add text
                    cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, status, (100, 150), font, 3, (0, 255, 0), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))

    return frame
# Custom CSS to style the page
st.markdown(
    """
    <style>
    .header {
        padding: 10px;
        background-color: #f0f0f0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #333333;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        color: #777777;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
#Create the Streamlit app
def main():
    st.title("Real-time Face Mask Detection")
    st.markdown('<div class="header">COVID 19 FaceMask Detector</div>', unsafe_allow_html=True)

    # Create a sidebar with app information
    st.sidebar.title("About")
    st.sidebar.info("This app uses a pre-trained model(MobileNet) to detect face masks in real-time using your webcam.")

    # Create a WebRTC video chat component
    webrtc_ctx = webrtc.webrtc_streamer(
        key="example",
        mode=webrtc.WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_transformer_factory=None,
        async_transform=True,
    )

    # Continuously capture frames from the camera until the "Stop" button is pressed
    while webrtc_ctx.video_receiver:
        # Get the latest frame from the video chat component
        frame = webrtc_ctx.video_receiver.frame

        if frame is not None:
            # Convert the frame to OpenCV format
            img = frame.to_ndarray(format="bgr24")

            # Perform face mask detection on the frame
            result_frame = detect_mask(img)

            # Display the frame with face mask detection
            st.image(result_frame, channels="BGR", caption="Face Mask Detection", use_column_width=True)
      # Footer
    st.markdown('<div class="footer">Created with ❤️ by Spandan Ghatak</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
