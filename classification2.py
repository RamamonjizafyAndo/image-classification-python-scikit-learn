import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import os
app = Flask(__name__)
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
img_height = 180
img_width = 180
model = tf.keras.models.load_model('my_model.keras')
sary = "rose.jpg"
label_image = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


def imgOptimization(dataImg):
    img = tf.keras.utils.load_img(
        dataImg, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    return img_array

@app.route('/img-classification', methods=['POST'])
def create():
    if 'img' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    img_data = request.files['img']
    if img_data.filename == '':
        return jsonify({'error': 'No selected image'}), 400
    input_dir = os.path.join(DATA_DIR, 'input')
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    image = Image.open(img_data)
    image.save(os.path.join(input_dir, img_data.filename))
    im = os.path.join(input_dir, img_data.filename)
    predictions = model.predict(imgOptimization(im))
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(label_image[np.argmax(score)], 100 * np.max(score))
    )

    return(label_image[np.argmax(score)])

if __name__ == '__main__':
    app.run(debug=True, port=5000)