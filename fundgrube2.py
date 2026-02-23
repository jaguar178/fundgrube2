import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import numpy as np
from PIL import Image
import tensorflow as tf
import json

# --- Modell laden ---
model_path = "./my_model/model.h5"
model = tf.keras.models.load_model(model_path)

with open("./my_model/metadata.json") as f:
    metadata = json.load(f)
class_names = metadata["labels"]

st.title("Teachable Machine Image Model mit Streamlit")

# --- Option: Webcam oder Upload wählen ---
option = st.radio("Bildquelle auswählen:", ("Webcam", "Bild hochladen"))

# --- Webcam Transformer ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.class_names = class_names

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_image()
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        class_idx = np.argmax(prediction[0])
        class_label = self.class_names[class_idx]
        probability = prediction[0][class_idx]

        # Text auf das Bild schreiben
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"{class_label}: {probability:.2f}", fill=(255, 0, 0))
        return np.array(img)

# --- Webcam Stream ---
if option == "Webcam":
    webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

# --- Bild Upload ---
elif option == "Bild hochladen":
    uploaded_file = st.file_uploader("Bild auswählen", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        # Bild für das Modell vorbereiten
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Vorhersage
        prediction = model.predict(img_array)
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {prediction[0][i]:.2f}")

        # Optional: Bestes Ergebnis hervorheben
        best_idx = np.argmax(prediction[0])
        st.success(f"Vorhergesagte Klasse: {class_names[best_idx]} ({prediction[0][best_idx]:.2f})")
