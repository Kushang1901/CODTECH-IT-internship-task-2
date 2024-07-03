import tensorflow as tf
import numpy as np
import urllib.request
from PIL import Image
# Load a pre-trained model (e.g., MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights="imagenet")
# Download an example image (you can replace this with your own image URL)
image_url = "https://example.com/path/to/your/image.jpg"
urllib.request.urlretrieve(image_url, "image.jpg")
# Load the image
image = Image.open("image.jpg")
image = image.resize((224, 224)) # Resize to match the model input size
image_array = np.array(image) / 255.0 # Normalize pixel values
# Make predictions
predictions = model.predict(np.expand_dims(image_array, axis=0))
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions.numpy())
# Print the top predicted labels
for _, label, confidence in decoded_predictions[0]:
print(f"{label}: {confidence:.2f}")