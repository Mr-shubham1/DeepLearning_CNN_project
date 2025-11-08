import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
# import tensorflow
# import keras
from tensorflow.keras.models import load_model
import time
import requests
import tempfile

# model = load_model("models/my_model_06.h5",compile=False);

url = 'https://huggingface.co/MRshubhamai/CNN-Model/resolve/main/my_model_06.h5'
response = requests.get(url, stream=True);
response.raise_for_status();

with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp:
    for chunk in response.iter_content(chunk_size=8192):
        temp.write(chunk)
    temp_path = temp.name 
model = load_model(temp_path,compile=False);

st.set_page_config(
    page_title="Convolutional neural network",
    page_icon="🤖",
    layout="wide"
)

col1,col2 = st.columns([7.5,2.5])
with col1:
    st.header("Gender, Age & Race Predicting AI Model (CNN)")
    st.caption("A Deep Learning AI model built using Convolutional Neural Networks")
with col2:
    st.image("gif/cnn layer gif.gif")
image_file = st.file_uploader("upload a face image",type=['jpg','jpeg','png']);

gender_labels = ['Male', 'Female']
race_labels = ['White', 'Black', 'Asian', 'Indian', 'Other'] 

if "result" not in st.session_state:
    st.session_state.result = None;

if image_file is not None:
    image = Image.open(image_file).convert('RGB');
    image = image.resize((128,128));
    col1,col2,col3 = st.columns([1,1,1])
    with col2:
        st.image(image, caption="Uploaded Image");
    image_tensor = np.array(image)/255.0;
    image_tensor = np.expand_dims(image_tensor,axis=0);


    col1,col2,col3 = st.columns([1,1,1]);
    with col2:
        if st.button("Run CNN Prediction"):
            with st.spinner("Running model, please wait..."):
                placeholder1 = st.empty()
                placeholder2 = st.empty()

                placeholder1.image("gif/cnn layer gif.gif", use_container_width=True)
                time.sleep(2.5)
                placeholder2.image("gif/ann layer gif.gif", use_container_width=True)
                time.sleep(2.5)
                placeholder1.empty()
                placeholder2.empty()
                st.success("✅ Model executed successfully! Predictions are ready below.")
                result = model.predict(image_tensor);
                st.session_state.result=result;
                

if st.session_state.result is not None:
    predicted_age = int(st.session_state.result[0][0])
    predicted_gender = gender_labels[int(st.session_state.result[1][0] > 0.5)]
    predicted_race = race_labels[np.argmax(st.session_state.result[2][0])]
    col1,col2,col3 = st.columns([1,1,1])
    with col1:
        st.markdown(f"<h4 style='text-align:center;'>Predicted Gender:<br>{predicted_gender}</h4>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h4 style='text-align:center;'>Predicted Age:<br>{predicted_age} years</h4>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<h4 style='text-align:center;'>Predicted Race:<br>{predicted_race}</h4>", unsafe_allow_html=True)

            

            

    