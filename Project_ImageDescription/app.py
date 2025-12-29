import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# ---------------- Load model dan tokenizer ----------------
model = load_model("model/model_final.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("model/max_length.pkl", "rb") as f:
    max_length = pickle.load(f)

# Load InceptionV3 untuk ekstraksi fitur
feature_model = InceptionV3(include_top=False, weights="imagenet", pooling="avg")

# ---------------- Fungsi Ekstrak Fitur ----------------
def extract_feature(img):
    img = img.resize((299, 299)).convert("RGB")
    x = np.expand_dims(np.array(img), axis=0)
    x = preprocess_input(x)
    feature = feature_model.predict(x, verbose=0)
    return feature.flatten()

# ---------------- Fungsi Generate Caption ----------------
def generate_caption(model, tokenizer, photo, max_length):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo.reshape(1,2048), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption

# ---------------- Streamlit UI ----------------
st.title("Image Captioning")
st.write("Upload gambar dan model akan men-generate captionnya.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    with st.spinner('Generating caption...'):
        feature = extract_feature(image)
        caption = generate_caption(model, tokenizer, feature, max_length)
    st.success('Caption generated!')
    st.write(caption)
