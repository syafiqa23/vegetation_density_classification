import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 380

model = tf.keras.models.load_model("../models/vegetation_model.h5")

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_data = test_gen.flow_from_directory(
    "../dataset/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)

pred = model.predict(test_data)

y_pred = np.argmax(pred, axis=1)
y_true = test_data.classes

print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=test_data.class_indices,
    yticklabels=test_data.class_indices
)

plt.title("Confusion Matrix")
plt.savefig("../outputs/confusion_matrix.png")
plt.show()