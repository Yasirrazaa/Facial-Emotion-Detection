# Importing Necessary Libraries
import cv2
import numpy as np
from keras.models import model_from_json
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, VideoTransformerBase
from PIL import Image
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

# Setting up GPU:
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)



st.cache_resource()
st.cache()

logo = Image.open('logo.png')
# Setting Page Title:
st.set_page_config(page_title="Emotion Detection" , page_icon=logo, initial_sidebar_state='auto')

# 2. horizontal menu
selected = option_menu(
    menu_title="",
    options=["Home", "About Us"], #required
    default_index=0, #optional
    orientation= "horizontal",
    )
css_example = '''                                           
    <link rel="stylesheet" href="font-awesome.css">    
    
    <style>
        .bodyP1{
            font-size: 20px;
        }
        .footer{
            display: flex;
            justify-content:center;
            align-items: center;
            font-size: 20px;
            font-weight: 300;
            margin-top: 50px;
        }
        .aboutUs p{
            font-size: 18px;
            text-align: justify;
        }
        .header{
            display: flex;
            flex-direction:column;
            justify-content: center;
            align-items: center;
        }
    </style>
'''
st.write(css_example, unsafe_allow_html=True)

# Declaring Classes
emotion_classes = {
    0: "Angry", 
    1: "Disgust", 
    2: "Fear", 
    3: "Happy", 
    4: "Neutral", 
    5: "Sad", 
    6: "Surprise"}

# # Loading Trained Model:
json_file = open(r"model_v3.json", 'r')

# Loading model.json file into model
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

 # Loading Weights:
model.load_weights(r"model_v3.h5")

print("Model Loaded Successfully")

# Loading Face Cascade
try: 
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.error("Unable to load Cascade Classifier", icon="⚠️")


class EmotionDetector(VideoProcessorBase):
    #@st.cache_data
    def transform(self, frame):
        # Converting frame into 2 array of RGB format.
        img = np.array(frame.to_ndarray(format = "bgr24"))

        #Converting the Captured frame to gray scale:
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces available on camera:
        num_face = face_detector.detectMultiScale(gray_frame, scaleFactor = 1.3, minNeighbors = 5)

        # Take each face available on the camera and preprocess it:
        for (x, y, w, h) in num_face:
            cv2.rectangle(img, (x,y-50), (x+w, y+h+10), (0,255,0), 4)
            roi_gray_frame = gray_frame[y:y+h, x: x+w]
            cropped_img = np.expand_dims(cv2.resize(roi_gray_frame, (48,48), -1), 0)

            #Predict the emotion:
            if np.sum([roi_gray_frame])!=0:
                emotion_prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                label_position = (x,y)
                output = str(emotion_classes[maxindex])
                cv2.putText(img,output,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(img,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        return img


def main():
    # Face Analysis Application #

    # Menu Cases:
    if selected == "Home":
        header = """
                    <div class = "header">
                        <h1>RED</h1>
                    </div>
                """
        st.markdown(header,unsafe_allow_html=True)
        html_temp_home1 = """
                            <div>
                                <p class="bodyP1">Welcome to RED (Realtime Emotion Detection) a web application that detect facial emotions in real-time. <br> Our application combines OpenCV and Convolutional Neural Networks to accurately detect the seven basic emotions anger , sad , surprise , happy , disgust , fear and neutral. </p>
                                <h2>Try Now</h2>
                                <p>Click on start to use webcam and detect your face emotion</p>
                            </div>
                            """
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        webrtc_streamer(key="emotion_detection", video_processor_factory=EmotionDetector, async_processing=True,audio_processor_factory=None)
        html_home2  = """
            <div class="footer">
                <p>&copy; 2023 by Yasir Raza,M Asim,Aleeha Amjad and Rafay Malik (Group 1). All rights reserved.</p>
            </div>
        """
        st.write(html_home2, unsafe_allow_html=True)
    elif selected == "About Us":
        st.title("About Us")
        html_temp_about1= """
                            <div class="aboutUs">
                                <p>
                                    RED (Realtime Emotion Detection) is an Web Application created by <b>Yasir Raza</b>,<b>Rafay Malik</b>, <b>Aleeha Amjad</b> and <b>M Asim</b> <b>(Group 1)</b> as our Psychology semester project. The application uses OpenCV and Convolutional Neural Network model to accurately detect human facial emotions in real-time.
                                </p>
                                <p>
                                    Together, we worked on this project to create an easy-to-use web application that can accurately detect a wide range of emotions. We hope to implement more features and improve the accuracy and processing time of the application in future.
                                </p>
                                <p>
                                    You can access the web application code, notebook and models at <a target="_blank" href="https://github.com/Yasirrazaa/Facial-Emotion-Detection.git">Facial Emotion Detection</a>.
                                </p>
                                <p>
                                    If you have any questions or feedback about this project, please feel free to contact us <br> <a target="_blank" href="https://www.linkedin.com/in/yasir-abdali-9b78a6229/"> Yasir Raza </a> or <a target="_blank" href="https://www.linkedin.com/in/muhammad-asim-725595250/">M Asim</a>
                                </p>
                            </div>
                                    """
        st.markdown(html_temp_about1, unsafe_allow_html=True)
        #st.title("Authors:")
        
    else:
        pass


if __name__ == "__main__":
    main()