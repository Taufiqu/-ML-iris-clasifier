from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("iris_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        data = [
            float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width'])
        ]
        prediction = model.predict([data])
        iris_classes = ['Setosa', 'Versicolor', 'Virginica']
        result = iris_classes[prediction[0]]
        return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)

import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)

