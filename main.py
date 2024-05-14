from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Load the TF-IDF vectorizer and models
tfidf_vectorizer = joblib.load('MLengineer-flash/src/models/tfidf_vectorizer.pkl')
logistic_regression_model = joblib.load('MLengineer-flash/src/models/category/Random Forest_model_category.joblib')
et_model = joblib.load('MLengineer-flash/src/models/emailtype/Neural Network_model_et.joblib')

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    # Vectorize the input text using the loaded TF-IDF vectorizer
    transformed_text = tfidf_vectorizer.transform([request.text])

    # Example: Using the logistic regression model for prediction
    category_pred = logistic_regression_model.predict(transformed_text)

    if category_pred[0] == 'category_1':
        email_type_prediction = 'email_type_138' #Hardcoded this because all the category_1 are mapped to class email_type_138, even without this the implementation will work fine but this saves time
    else:
        email_type_prediction = et_model.predict(transformed_text)[0]

    return {"category": category_pred[0], "email_type": email_type_prediction}