import gradio as gr
import numpy as np
import cv2
import pickle

# Load your model
with open("sign_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define labels for output (update this according to your model)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Prediction function
def predict_sign_from_image(image):
    if image is None:
        return "No image"
    
    # Preprocess image: resize, grayscale, flatten — adjust based on training
    img_resized = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten().reshape(1, -1)

    prediction = model.predict(flat)[0]
    return f"Predicted Sign: {labels[prediction]}"

# Gradio interface
demo = gr.Interface(
    fn=predict_sign_from_image,
    inputs=gr.Image(source="webcam", streaming=True, label="Show Your Sign"),
    outputs=gr.Text(label="Detected Letter"),
    live=True,
    title="Real-Time Sign Language Detector"
)

demo.launch()