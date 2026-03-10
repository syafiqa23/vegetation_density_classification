import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

import os
import gdown
import subprocess
import sys

try:
    import tensorflow as tf
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.13.0"])
    import tensorflow as tf
# ======================
# CONFIG
# ======================

IMG_SIZE = 380
MODEL_PATH = "models/vegetation_model.h5"
MODEL_PATH = "models/vegetation_model.h5"

FILE_ID = "1EWYS_6yCOmIdzLRvLtoLrpSpmOw7qKCu"

if not os.path.exists(MODEL_PATH):

    os.makedirs("models", exist_ok=True)

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(url, MODEL_PATH, quiet=False)

classes = ["bare","heavily_grazed","softly_grazed"]

# ======================
# DESKRIPSI DATASET
# ======================

class_description = {
    "bare": "Area dengan vegetasi sangat sedikit atau hampir tidak ada. Permukaan tanah terlihat jelas dan vegetasi jarang ditemukan.",
    
    "heavily_grazed": "Area dengan kepadatan vegetasi tinggi. Citra drone menunjukkan vegetasi yang rapat dan menutupi sebagian besar permukaan tanah.",
    
    "softly_grazed": "Area dengan kepadatan vegetasi sedang. Vegetasi masih terlihat tetapi tidak terlalu rapat."
}

# ======================
# LOAD MODEL
# ======================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ======================
# GRADCAM FUNCTION
# ======================

def make_gradcam(img_array, model, last_conv_layer="top_conv"):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)

    return heatmap


# ======================
# STREAMLIT UI
# ======================

st.title("🌿 Vegetation Density Classification")
st.write("Sistem klasifikasi kepadatan vegetasi menggunakan **Convolutional Neural Network (CNN)** dengan visualisasi **Explainable AI Grad-CAM**.")

st.info(
"""
Grad-CAM digunakan untuk menjelaskan keputusan model AI.  
Area **merah** menunjukkan bagian gambar yang paling diperhatikan model saat menentukan kelas vegetasi.
"""
)

uploaded = st.file_uploader("Upload Drone Image", type=["jpg","png","jpeg"])

if uploaded:

    img = Image.open(uploaded).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ======================
    # PREPROCESS
    # ======================

    img_resized = img.resize((IMG_SIZE,IMG_SIZE))
    img_array = np.array(img_resized)

    img_array = np.expand_dims(img_array,axis=0)
    img_array = preprocess_input(img_array)

    # ======================
    # PREDICTION
    # ======================

    prediction = model.predict(img_array)

    pred_index = np.argmax(prediction)
    pred_class = classes[pred_index]
    confidence = prediction[0][pred_index]

    # ======================
    # OUTPUT
    # ======================

    st.subheader("Prediction Result")

    col1,col2 = st.columns(2)

    with col1:

        st.metric("Class",pred_class)
        st.metric("Confidence",f"{confidence:.2f}")

        st.progress(float(confidence))

        # ======================
        # DESKRIPSI KELAS
        # ======================

        st.markdown("### Deskripsi Kelas")
        st.write(class_description[pred_class])

    # ======================
    # CONFIDENCE CHART
    # ======================

    with col2:

        st.subheader("Class Probability")

        fig, ax = plt.subplots()

        ax.bar(classes,prediction[0])

        ax.set_ylabel("Confidence")
        ax.set_ylim(0,1)

        st.pyplot(fig)

    # ======================
    # GRADCAM
    # ======================

    st.subheader("GradCAM Visualization")

    heatmap = make_gradcam(img_array,model)

    heatmap = cv2.resize(heatmap,(IMG_SIZE,IMG_SIZE))
    heatmap = np.uint8(255*heatmap)

    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    original = cv2.cvtColor(np.array(img_resized),cv2.COLOR_RGB2BGR)

    superimposed = cv2.addWeighted(original,0.6,heatmap,0.4,0)

    st.image(
        cv2.cvtColor(superimposed,cv2.COLOR_BGR2RGB),
        caption="GradCAM Result",
        use_container_width=True
    )

    # ======================
    # PENJELASAN GRADCAM
    # ======================

    st.markdown("### Interpretasi GradCAM")

    st.write(
    """
    Visualisasi Grad-CAM menunjukkan area pada gambar yang paling mempengaruhi keputusan model CNN.
    
    🔴 **Merah** → Area paling penting bagi model  
    🟠 **Kuning** → Area cukup penting  
    🔵 **Biru** → Area kurang berpengaruh
    
    Jika area vegetasi pada citra ditandai warna merah atau kuning, maka model benar-benar
    memfokuskan perhatian pada vegetasi saat melakukan klasifikasi.
    """
    )