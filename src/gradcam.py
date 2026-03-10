import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================
# CONFIG
# ==========================
IMG_SIZE = 380
MODEL_PATH = "../models/vegetation_model.h5"
IMG_PATH = "../dataset/test/bare/1.png"
OUTPUT_PATH = "../outputs/gradcam_result.png"

class_names = ["bare", "heavily_grazed", "softly_grazed"]

# ==========================
# LOAD MODEL
# ==========================
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# LOAD IMAGE
# ==========================
img = image.load_img(IMG_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# ==========================
# PREDICTION
# ==========================
preds = model.predict(img_array)
class_idx = np.argmax(preds[0])
confidence = preds[0][class_idx]

print("Prediction:", class_names[class_idx])
print("Confidence:", confidence)

# ==========================
# GRADCAM
# ==========================
last_conv_layer = "top_conv"

grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer).output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, class_idx]

grads = tape.gradient(loss, conv_outputs)

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0].numpy()
pooled_grads = pooled_grads.numpy()

for i in range(pooled_grads.shape[-1]):
    conv_outputs[:, :, i] *= pooled_grads[i]

heatmap = np.mean(conv_outputs, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# ==========================
# RESIZE HEATMAP
# ==========================
heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
heatmap = np.uint8(255 * heatmap)

# ==========================
# APPLY COLORMAP
# ==========================
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# ==========================
# LOAD ORIGINAL IMAGE
# ==========================
original_img = cv2.imread(IMG_PATH)
original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

# ==========================
# OVERLAY HEATMAP
# ==========================
superimposed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

# ==========================
# SAVE RESULT
# ==========================
cv2.imwrite(OUTPUT_PATH, superimposed)

print("GradCAM saved to:", OUTPUT_PATH)

# ==========================
# DISPLAY
# ==========================
plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
plt.title(f"{class_names[class_idx]} ({confidence:.2f})")
plt.axis("off")
plt.show()