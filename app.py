import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("model.h5")

# --- Preprocessing & GradCAM Functions (Same as before) ---
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image / 255.0
    return image

def get_gradcam(image_array, model, last_conv_layer_name="conv5_block3_out"):
    img_tensor = tf.expand_dims(image_array, axis=0)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (image_array.shape[1], image_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image_array, 0.6, heatmap, 0.4, 0)
    return superimposed_img

# -------------------- Streamlit UI --------------------

# Page settings
st.set_page_config(page_title="Brain Tumor GradCAM", layout="wide")

# Header
st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor MRI - GradCAM Visualization</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload a brain MRI image to see model predictions and Grad-CAM heatmaps for visual explanation.</p>", unsafe_allow_html=True)
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "png", "jpeg"])

# When image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="🖼 Original MRI Image", use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image)
    gradcam_result = get_gradcam((img_array * 255).astype("uint8"), model)

    with col2:
        st.image(gradcam_result, caption="🔥 Grad-CAM Heatmap", use_column_width=True)

    # Model prediction
    pred = model.predict(tf.expand_dims(img_array, axis=0))
    classes = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']  # Replace as needed
    prediction = classes[np.argmax(pred)]
    confidence = 100 * np.max(pred)

    st.markdown("---")
    st.markdown("### 🧬 Model Prediction:")
    st.success(f"{prediction}** with *{confidence:.2f}%* confidence")