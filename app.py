
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json

app = Flask(__name__)

df = pd.read_csv("dataset/Mumbai House Prices.csv")

model = joblib.load("modal/mumbai_house_price_model.pkl")

with open("model_columns.json", "r") as file:
    model_columns = json.load(file)

bhk_options = sorted(df["bhk"].dropna().unique().tolist())
type_options = sorted(df["type"].dropna().unique().tolist())
age_options = sorted(df["age"].dropna().unique().tolist())
region_options = sorted(df["region"].dropna().unique().tolist())

def get_options():
    return {
        "bhk_options": bhk_options,
        "type_options": type_options,
        "age_options": age_options,
        "region_options": region_options
    }

@app.route("/")
def home():
    return render_template("index.html", **get_options())

@app.route("/predict", methods=["POST"])
def predict():
    try:
        bhk = float(request.form["bhk"])
        area = float(request.form["area"])
        property_type = request.form["type"]
        age = request.form["age"]
        region = request.form["region"]

        if bhk not in bhk_options:
            raise ValueError("Invalid BHK selected")

        if property_type not in type_options:
            raise ValueError("Invalid property type")

        if age not in age_options:
            raise ValueError("Invalid property age")

        if region not in region_options:
            raise ValueError("Invalid region")

        if area <= 0:
            raise ValueError("Area must be greater than zero")

        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=model_columns
        )

        input_data["bhk"] = bhk
        input_data["area"] = area

        features = [
            f"type_{property_type}",
            f"age_{age}",
            f"region_{region}"
        ]

        for feature in features:
            if feature in input_data.columns:
                input_data[feature] = 1

        prediction = float(model.predict(input_data)[0])

        return jsonify({
            "success": True,
            "prediction": round(prediction, 2),
            "bhk": bhk,
            "type": property_type,
            "age": age,
            "region": region,
            "area": area
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True)
