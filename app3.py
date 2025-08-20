import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras

# App title
st.title("🎤 Scream Detection using ANN Model")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Load trained ANN model
    model = keras.models.load_model("ann_model.h5")

    # Preprocess audio
    y, sr = librosa.load(uploaded_file, sr=22050)  # resample to 22.05kHz
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract 40 MFCCs
    mfcc_scaled = np.mean(mfcc.T, axis=0)  # Take mean across time

    # Reshape for ANN model input
    X = np.expand_dims(mfcc_scaled, axis=0)

    # Prediction
    prediction = model.predict(X)
    pred_class = np.argmax(prediction, axis=1)

    # Display result
    if pred_class[0] == 1:  # Assuming class 1 = scream
        st.error("⚠️ Scream Detected!")
    else:  # Assuming class 0 = normal
        st.success("✅ No Scream Detected")
