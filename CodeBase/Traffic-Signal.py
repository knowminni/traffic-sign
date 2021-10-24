import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from PIL import Image


if __name__ == '__main__':

    model = load_model('traffic_classifier.h5')

    #Dictionary to Label all traffic signs class.
    classes = {
                1:'Speed limit - 20km/h',
                2:'Speed limit - 30km/h', 
                3:'Speed limit - 50km/h', 
                4:'Speed limit - 60km/h', 
                5:'Speed limit - 70km/h', 
                6:'Speed limit - 80km/h', 
                7:'End of speed limit - 80km/h', 
                8:'Speed limit - 100km/h', 
                9:'Speed limit - 120km/h', 
                10:'No Passing', 
                11:'No Passing for Vehicles of over 3.5 tons', 
                12:'Right-of-way at Intersection', 
                13:'Priority Road', 
                14:'Yield', 
                15:'Stop', 
                16:'No Vehicles', 
                17:'Vehicles of more than 3.5 tons are Prohibited', 
                18:'No Entry', 
                19:'General Caution', 
                20:'Dangerous Curve Left', 
                21:'Dangerous Curve Right', 
                22:'Double Curve', 
                23:'Bumpy Road', 
                24:'Slippery Road', 
                25:'Road Narrows on the Right', 
                26:'Road at Work', 
                27:'Traffic Signals', 
                28:'Pedestrians', 
                29:'Children Crossing', 
                30:'Bicycles Crossing', 
                31:'Beware of Ice/Snow',
                32:'Wild Animals Crossing', 
                33:'End of Speed + Passing Limits', 
                34:'Turn Right Ahead', 
                35:'Turn Left Ahead', 
                36:'Ahead Only', 
                37:'Go Straight or Right', 
                38:'Go Straight or Left', 
                39:'Keep Right', 
                40:'Keep Left', 
                41:'Roundabout Mandatory', 
                42:'End of No Passing', 
                43:'End of No Passing for Vehicles of more than 3.5 tons' 
            }



    c1 = st.container()
    c1.title('Traffic Signal Recognition')
    c1.subheader('Using Deep Learning and Computer Vision')
    c1.write('Project by Neha Vishwakarma, MCA 3rd Semester, GNIT')

    c2 = st.container()
    c2.header('Let\'s try out!')
    imageFile = c2.file_uploader('Choose an Image File', accept_multiple_files=False, type=['png','jpg'], help = 'Pick a PNG/JPG file')

    if imageFile is not None:
        img = Image.open(imageFile)
        st.image(img, caption='Image Uploaded.', use_column_width='auto')
        st.write("")

    if st.button('Recognise'):
        st.write('Classifying Image')
        st.write('Predicting Labels')
        img = img.resize((30,30))
        img = np.expand_dims(img, axis=0)
        img2 = np.array(img)
        pred = model.predict([img2])
        classpred = np.argmax(pred, axis = 1)
        sign = classes[int(classpred)+1]

        
        st.subheader("_Model Prediction_")
        st.write("This Sign is: ")
        st.title(sign)
        st.balloons()

        
        st.write("")
        st.write("")
        st.sidebar.write("**_Developed by Neha Vishwakarma_**")
        st.sidebar.write("_Under the guidance of Mrs. Somalina Chowdhury, Asstt. Professor, Dept CA, GNIT_")
        st.sidebar.write('For completion of Minor Project for MCA-20 : 3rd Semester;')
        st.sidebar.subheader('Department of Computer Applications, Guru Nanak Institute of Technology')
        st.write("")
        st.write("")
        st.write("")
        st.write("**Hosted at GitHub:  @knowminni**")
        st.write("_Under Apache License 2.0_")
