from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the "memory"
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, 
                  age: int = Form(...), 
                  sex: str = Form(...), 
                  bp: str = Form(...), 
                  chol: str = Form(...), 
                  na_to_k: float = Form(...)):
    
    # Encode categorical inputs using saved encoders
    sex_enc = encoders['Sex'].transform([sex])[0]
    bp_enc = encoders['BP'].transform([bp])[0]
    chol_enc = encoders['Cholesterol'].transform([chol])[0]
    
    # Prepare features
    features = np.array([[age, sex_enc, bp_enc, chol_enc, na_to_k]])
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction_idx = model.predict(features_scaled)[0]
    prediction = encoders['Drug'].inverse_transform([prediction_idx])[0]
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "prediction": prediction
    })