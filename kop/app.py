import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model
model = tf.keras.models.load_model("CNN_Model64.h5")

# Define image size and class labels
image_size = (64, 64)
diseases_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca', 'Grape___Leaf_blight', 'Grape___healthy',
    'Orange___Citrus_greening', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper___Bacterial_spot', 'Pepper___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 'Tomato___Target_Spot',
    'Tomato___Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Prediction function
def predict_image(image_path, confidence_threshold=0.5):
    # Load and preprocess the image
    image = load_img(image_path, target_size=image_size)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array)
    sorted_indices = np.argsort(predictions[0])[::-1]
    max_confidence = predictions[0][sorted_indices[0]]
    predicted_label = diseases_labels[sorted_indices[0]]

    # Check if it's a plant or not
    if max_confidence < confidence_threshold:
        return "Not a Plant", max_confidence
    return predicted_label, max_confidence

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        # Get uploaded image
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Make prediction
            label, confidence = predict_image(filepath)

            return render_template("index.html", image_path=filepath, label=label, confidence=f"{confidence*100:.2f}%")

    return render_template("index.html", image_path=None, label=None, confidence=None)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
