from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# --------------------------------------------------
# Load the saved ML model (must be in the same folder)
# --------------------------------------------------
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# --------------------------------------------------
# Initialize FastAPI app
# --------------------------------------------------
app = FastAPI()

# --------------------------------------------------
# Define input schema using Pydantic
# --------------------------------------------------
class ChurnInput(BaseModel):
    tenure: float
    MultipleLines: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

# --------------------------------------------------
# Root endpoint (optional)
# --------------------------------------------------
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(data: ChurnInput):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Predict using the loaded pipeline
    pred = model.predict(df)[0]

    return {"churn_prediction": int(pred)}
