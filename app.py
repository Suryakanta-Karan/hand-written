import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the pre-trained digit recognition model (Random Forest, for example)
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier

mnist = fetch_openml("mnist_784")
X, y = mnist.data, mnist.target
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Create a function to process and predict digits
def process_and_predict_digit(image):
    try:
        # Load and preprocess the image
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img.reshape(1, -1)

        # Predict the digit
        predicted_digit = model.predict(img)[0]

        return predicted_digit

    except Exception as e:
        print(str(e))
        return -1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare_digits', methods=['POST'])
def compare_digits():
    try:
        # Receive two images
        image1 = request.files['image1']
        image2 = request.files['image2']

        if image1 and image2:
            digit1 = process_and_predict_digit(image1)
            digit2 = process_and_predict_digit(image2)

            if digit1 == digit2:
                return jsonify({'result': True})
            else:
                return jsonify({'result': False})

    except Exception as e:
        print(str(e))
        return jsonify({'result': False})

if __name__ == '__main__':
    app.run(debug=True)
