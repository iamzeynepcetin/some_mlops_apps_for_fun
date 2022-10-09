import requests

test = {"Pregnancies": 6.0,
        "Glucose": 148.0,
        "BloodPressure": 72.0,
        "SkinThickness": 35.0,
        "Insulin": 0.0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50.0
        }

url= 'http://localhost:9696/predict'
response= requests.post(url, json = test)
print(response.json())


#preprocessed = predict.preprocess(test)
#pred =predict.predict(preprocessed)
#print(pred)