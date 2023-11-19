from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello_world():  
    return "<p>Hello, World!</p>"

@app.route("/", methods=["POST"])
def hello_world_post():    
    return {"op" : "Hello, World POST " + request.json["suffix"]}

#LOgic Started.

def predict_digit(image_data):
    # Example: Assuming if sum of pixel values > 5, predict as '1', else predict as '0'
    pixel_sum = sum(sum(row) for row in image_data)
    return 1 if pixel_sum > 5 else 0

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("input")
    predicted_digit = predict_digit(data)
    return str(predicted_digit)

def test_client():
    return app.test_client()

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=8081 , debug=True)