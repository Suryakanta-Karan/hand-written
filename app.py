import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from skimage import measure

app = Flask(__name__)

# Helper function to compute structural similarity index between two images
def compare_images(image1, image2):
    # Convert the images to grayscale
    image1 = image1.convert('L')
    image2 = image2.convert('L')

    # Convert images to numpy arrays
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # Compute the Structural Similarity Index (SSIM)
    ssim = measure.compare_ssim(image1_array, image2_array)

    return ssim

# API route to check if two images are similar
@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    try:
        # Check if two image files are provided in the request
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({"error": "Two image files are required."}), 400

        image1 = request.files['image1']
        image2 = request.files['image2']

        if image1.filename == '' or image2.filename == '':
            return jsonify({"error": "Image filenames are empty."}), 400

        image1 = Image.open(image1)
        image2 = Image.open(image2)

        # Compare the images for similarity
        similarity = compare_images(image1, image2)

        if similarity > 0.7:  # Adjust the threshold as needed
            return jsonify({"result": True, "similarity": similarity})
        else:
            return jsonify({"result": False, "similarity": similarity})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
