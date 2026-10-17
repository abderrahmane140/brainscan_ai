from pathlib import Path
import json

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf


# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models/best_model.keras"
LABELS_PATH = PROJECT_ROOT / "data/processed/labels.json"

IMAGE_SIZE = (224, 224)


# Streamlit page
st.set_page_config(
    page_title="BrainScan AI",
    page_icon="🧠",
    layout="centered"
)


# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

with open(LABELS_PATH, "r") as file:
    label_mapping = json.load(file)


# Predict an MRI image
def predict_image(uploaded_file):

    image_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Invalid image file.")

    # Convert BGR to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize the image
    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    # Normalize pixel values
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    # Make prediction
    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    prediction = label_mapping[str(predicted_class)]
    confidence = float(probabilities[predicted_class])

    return prediction, confidence


# Streamlit interface
st.title("🧠 BrainScan AI")

st.write(
    "Upload a brain MRI image to classify it into one "
    "of four categories."
)

st.info(
    "Classes: Glioma, Meningioma, No Tumor, and Pituitary."
)

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png", "bmp"]
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded MRI Image",
        width=300
    )

    if st.button("Predict", type="primary"):

        try:
            prediction, confidence = predict_image(
                uploaded_file
            )

            st.subheader("Prediction Result")

            # Normalize class name for comparison
            prediction_key = (
                prediction.strip()
                .lower()
                .replace(" ", "")
                .replace("_", "")
            )

            # No tumor: green result
            if prediction_key == "notumor":

                st.success("Prediction: No Tumor")

                st.write("Predicted class: notumor")

            # Tumor class: red result
            else:

                tumor_names = {
                    "glioma": "Glioma tumor",
                    "meningioma": "Meningioma tumor",
                    "pituitary": "Pituitary tumor"
                }

                tumor_type = tumor_names.get(
                    prediction_key,
                    prediction.title()
                )

                st.error(
                    "The model predicted a tumor class."
                )

                st.write(f"Tumor type: {tumor_type}")

            st.write(
                f"Model confidence: {confidence:.2%}"
            )

        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )


st.divider()

st.caption(
    "Educational AI demonstration only. "
    "Predictions are not medical diagnoses. "
    "Consult a qualified medical professional "
    "for medical interpretation."
)