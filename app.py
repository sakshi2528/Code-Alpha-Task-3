from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("sales_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return "Sales Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([data["features"]])
    prediction = model.predict(features)
    return jsonify({"predicted_sales": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
