import os
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout, add
from tensorflow.keras.models import Model

DATASET_DIR = "dataset"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- load captions ----------------
def load_captions():
    path = os.path.join(DATASET_DIR, "captions.xlsx")
    df = pd.read_excel(path)
    df.columns = df.columns.str.lower()

    df["caption"] = df["caption"].astype(str)

    return df

def clean_caption(text):
    text = text.lower().strip()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return f"startseq {text} endseq"

# -------------- load feature extractor ------------
def extract_feature(img_path, model):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((299, 299))
    x = np.expand_dims(np.array(img), axis=0)
    x = preprocess_input(x)
    feat = model.predict(x, verbose=0)
    return feat.flatten()

# -------------- encode model --------------------
def define_model(vocab_size, max_len):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation="relu")(fe1)

    inputs2 = Input(shape=(max_len,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder = add([fe2, se3])
    decoder = Dense(256, activation="relu")(decoder)
    outputs = Dense(vocab_size, activation="softmax")(decoder)

    model = Model([inputs1, inputs2], outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model

# -------------------- MAIN ----------------------
df = load_captions()
df["cleaned"] = df["caption"].apply(clean_caption)

captions = df["cleaned"].tolist()
filenames = df["filename"].tolist()

extractor = InceptionV3(include_top=False, weights="imagenet", pooling="avg")

features = []
clean_caps = []

img_dir = os.path.join(DATASET_DIR, "images")

for i, fname in enumerate(filenames):
    img_path = os.path.join(img_dir, fname)

    if not os.path.exists(img_path):
        print("Missing:", img_path)
        continue

    feat = extract_feature(img_path, extractor)
    features.append(feat)
    clean_caps.append(captions[i])

features = np.array(features)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_caps)

with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in clean_caps)

with open(os.path.join(MODEL_DIR, "max_length.pkl"), "wb") as f:
    pickle.dump(max_length, f)

# create training sequences
X1, X2, y = [], [], []

for i, cap in enumerate(clean_caps):
    seq = tokenizer.texts_to_sequences([cap])[0]

    for j in range(1, len(seq)):
        in_seq, out = seq[:j], seq[j]

        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out = tf.keras.utils.to_categorical(out, num_classes=vocab_size)

        X1.append(features[i])
        X2.append(in_seq)
        y.append(out)

X1 = np.array(X1)
X2 = np.array(X2)
y = np.array(y)

model = define_model(vocab_size, max_length)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, "model_best.h5"), 
    monitor="loss", 
    save_best_only=True
)

model.fit([X1, X2], y, epochs=20, batch_size=32, callbacks=[checkpoint])

model.save(os.path.join(MODEL_DIR, "model_final.h5"))

print("Training completed!")
