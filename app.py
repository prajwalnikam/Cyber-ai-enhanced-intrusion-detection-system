from flask import Flask, render_template, request
import numpy as np
from joblib import load

app = Flask(__name__)
model = load("random_forest_model.joblib")  # Ensure this file is in the same folder

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # For testing: use 83 dummy features with the same value
        input_data = np.array([ [1.0]*83 ])  # Replace 1.0 with your test values
        prediction = model.predict(input_data)[0]
        return render_template("index.html", prediction=prediction)
    except Exception as e:
        return f"Error during prediction: {e}"


if __name__ == "__main__":
    app.run(debug=True)