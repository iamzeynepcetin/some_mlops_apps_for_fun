
import pickle
import pandas as pd
from flask import Flask, request, jsonify


model = pickle.load(open("DEPLOYMENT/diabet_model.pkl",'rb'))

def preprocess(test):
    new_feature = test["Glucose"] + test["BloodPressure"]
    test["new_feature"] = new_feature
    return test


def predict(X):
    preds = model.predict([pd.Series(X)])
    return float(preds[0])


app = Flask('diabet-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    test = request.get_json()
    preprocessed = preprocess(test)
    pred= predict(preprocessed)

    result = {
        'is_diabet': pred
    }
    print(result)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host = '0.0.0.0', port=9696)

"""
It can run from terminal with this command: gunicorn --bind= 0.0.0.0:9696 predict:app
"""