import streamlit as st
import tensorflow as tf
import numpy as np


# Load the trained model
model = tf.keras.models.load_model("ann_model.h5")


# Title for the web app
st.title("ANN Prediction App")


# Input fields for x1 and x2
x1 = st.number_input("Enter x1 (numeric value):", value=0.0, step=1.0)
x2 = st.number_input("Enter x2 (numeric value):", value=0.0, step=1.0)


# Predict button
if st.button("Predict"):
    # Prepare the input data for the model
    input_data = np.array([[x1, x2]])
    prediction = model.predict(input_data)
    # Convert prediction to binary output (0 or 1)
    output = 1 if prediction[0][0] > 0.5 else 0
    # Display the result
    st.success(f"The predicted output is: {output}")



