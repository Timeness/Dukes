import json, pickle, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def train_model():
    with open("model/dataset.json", "r") as f:
        data = json.load(f)

    X = [x["text"] for x in data]
    y = [x["intent"] for x in data]

    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)

    clf = LinearSVC()
    clf.fit(X_vect, y)

    with open("model/intent_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("[+] Model trained and saved.")
