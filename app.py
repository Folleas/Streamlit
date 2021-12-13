import streamlit as st
from tensorflow import keras
import cv2
import numpy as np
model_new = keras.models.load_model('./NumberGuess.model')

st.title("MNIST Digit Recognizer")

SIZE = 192

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred = model_new.predict(test_x.reshape(1, 28, 28, 1))
    st.write(f'result: {np.argmax(pred[0])}')
