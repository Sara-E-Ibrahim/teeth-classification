import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("teeth_model.h5")

model = load_model()

IMG_SIZE = (128, 128)   
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']


st.title("🦷 Oral Disease Classification App")
st.markdown("Developed by **Sara Essam** ❤️")

uploaded_file = st.file_uploader(
    "Upload an oral image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize(IMG_SIZE)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = np.max(prediction)

    st.success(f"Prediction: {predicted_label}")
    st.info(f"Confidence: {confidence * 100:.2f}%")
