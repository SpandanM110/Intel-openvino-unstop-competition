from flask import Flask, render_template, request, redirect, flash, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import secrets
import tensorflow as tf

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Set a random secret key

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a function to preprocess the uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = img.convert("RGB")
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)
    return img

# Define a function to classify the uploaded image
def classify_image(image_path):
    try:
        img = preprocess_image(image_path)
        model = tf.keras.models.load_model('custom_yolo_model.h5')  # Load the model
        prediction = model.predict(img)
        predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the index of the predicted class
        CLASS_LABELS = ["Crop", "Weed"]  # Update with your class labels
        predicted_class = CLASS_LABELS[predicted_class_index]
        return predicted_class
    except Exception as e:
        return f'Image classification error: {str(e)}'

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No image file uploaded.')
        return redirect(request.url)

    image_file = request.files['image']

    if image_file.filename == '':
        flash('No selected image file.')
        return redirect(request.url)

    if image_file:
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

        # Classify the uploaded image
        predicted_class = classify_image(image_path)

        return render_template('result.html', image_path=image_path, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
