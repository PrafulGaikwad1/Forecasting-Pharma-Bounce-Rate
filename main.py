from fastapi import FastAPI, HTTPException
import uvicorn
import pickle
from pydantic import BaseModel
from datetime import datetime
import pandas as pd

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello"}

@app.get("/home")
async def home():
    return {"Hello": "Welcome to our store!"}

class PredictionRequest(BaseModel):
    drug_name: str
    start_month: str
    end_month: str

# Load the ARIMA models from pickle files
with open('drug_1.pkl', 'rb') as file:
    model1 = pickle.load(file)

with open('drug_2.pkl', 'rb') as file:
    model2 = pickle.load(file)

with open('drug_3.pkl', 'rb') as file:
    model3 = pickle.load(file)

with open('drug_4.pkl', 'rb') as file:
    model4 = pickle.load(file)

with open('drug_5.pkl', 'rb') as file:
    model5 = pickle.load(file)

@app.get("/predict/{drug_name}/{start}/{end}")
def predict(drug_name: str, start_month: str, end_month: str):
    try:
        # Load the corresponding ARIMA model based on drug_name
        if drug_name == 'SODIUM CHLORIDE IVF 100ML':
            model = model1
        elif drug_name == 'SEVOFLURANE 99.97%':
            model = model2
        elif drug_name == 'SODIUM CHLORIDE 0.9%':
            model = model3
        elif drug_name == 'ONDANSETRON 2MG/ML':
            model = model4
        elif drug_name == 'MULTIPLE ELECTROLYTES 500ML IVF':
            model = model5
        else:
            return {"error": "We don,t have this Drug"}
        
        # Convert start_month and end_month strings to datetime objects
        start_date = datetime.strptime(start_month, "%Y-%m-%d")
        end_date = datetime.strptime(end_month, "%Y-%m-%d")
        
        # Define date_range based on start and end parameters (you might want to parse these as dates)
        # Ensure that the date_range corresponds to the model's index
        date_range = pd.date_range(start=model.fittedvalues.index[-1], periods=(end_date - start_date).days + 1, freq='D')
        
        # Predict for the specified date range
        predictions = model.predict(start=date_range[0], end=date_range[-1], dynamic=True).tolist()
        
        # Create a response dictionary 
        return {
            "drug_name": drug_name,
            "start_month": start_month,
            "end_month": end_month,
            "predictions": predictions
            }
    
    except Exception as e:
        return{"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)