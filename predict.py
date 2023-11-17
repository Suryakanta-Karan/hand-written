from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = '/home/suryakantak/hand-written/app/model/model.pkl'
try:
    trained_model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {str(e)}")
    trained_model = None

def compare_images(image1, image2):
    # Convert input strings to NumPy arrays
    image1_array = np.array(image1, dtype=float)
    image2_array = np.array(image2, dtype=float)

    # Flatten the arrays to 1D
    image1_array = image1_array.flatten()
    image2_array = image2_array.flatten()

    # Use the trained model to predict if the images are the same or different
    prediction = trained_model.predict([image1_array, image2_array])

    return prediction[0]


@app.route('/predict', methods=['POST'])
def compare_images_endpoint():
    try:
        print("Request Received")
        data = request.get_json()
        
        image1 = data['image1']
        image2 = data['image2']

        result = compare_images(image1, image2)
        

        response = {'result': 'Same' if result == 1 else 'Different'}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/')
def home():
    return 'Hello, ML Model App!'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)#, #debug=True)
    