import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from PIL import Image


st.title("Number Predictor! ")

uploaded_file = st.file_uploader(label="Upload image of a number herrreee",accept_multiple_files=False)
model = joblib.load("../models/kap_4_15_model.joblib")
scaler = joblib.load("../models/kap_4_15_scaler.joblib")
img = None
if uploaded_file != None:
    img = Image.open(uploaded_file).convert("L").resize((28,28))
    img_arr = np.array(img)
    if img_arr.mean() > 127:
        img_arr = 255 - img_arr
    img_arr = img_arr.reshape(784)
    img_arr = img_arr.reshape(-1,1).T
    img_arr = scaler.transform(img_arr)
    st.subheader(model.predict(img_arr)[0]) 