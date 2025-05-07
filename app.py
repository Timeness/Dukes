from flask import Flask, request, render_template, jsonify
import pickle, json, os
from train import train_model

app = Flask(__name__)

dataset_path = "model/dataset.json"
jailbreak_words = ["godmode", "dan", "root", "override"]

def load_model():
    with open("model/intent_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/add", methods=["POST"])
def add_data():
    text = request.form["text"]
    intent = request.form["intent"]

    if not os.path.exists(dataset_path):
        with open(dataset_path, "w") as f:
            json.dump([], f)

    with open(dataset_path, "r") as f:
        data = json.load(f)

    data.append({"text": text, "intent": intent})

    with open(dataset_path, "w") as f:
        json.dump(data, f, indent=2)

    return "Training data added."

@app.route("/train_model", methods=["GET"])
def retrain():
    train_model()
    return "Model retrained."

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]

    # Jailbreak check
    if any(word in text.lower() for word in jailbreak_words):
        return jsonify({"intent": "jailbreak_mode", "response": "Unfiltered GodMode Access Granted!"})

    model, vectorizer = load_model()
    vect = vectorizer.transform([text])
    intent = model.predict(vect)[0]
    return jsonify({"intent": intent, "response": f"Intent matched: {intent}"})
