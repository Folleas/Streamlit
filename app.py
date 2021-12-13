import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
from io import StringIO ## for Python 3

model_new = keras.models.load_model('./FashionGuess.model')

st.title("Fashion-MNIST Recognizer")

SIZE = 192

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
     # To read file as bytes:
     bytes_data = uploaded_file.getvalue()
     st.write(bytes_data)

     # To convert to a string based IO:
     stringio = StringIO(uploaded_file.getvalue().decode('latin-1'))
     st.write(stringio)

     # To read file as string:
     string_data = stringio.read()
     st.write(string_data)

     # Can be used wherever a "file-like" object is accepted:
     dataframe = pd.read_csv(uploaded_file)
     st.write(dataframe)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred = model_new.predict(test_x.reshape(1, 28, 28, 1))
    st.write(f'result: {np.argmax(pred[0])}')
    st.bar_chart(pred[0])
