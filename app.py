from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

required_files = [
    "regression_model.joblib",
    "classification_model.joblib",
    "iris_target_names.joblib",
]
missing = [name for name in required_files if not (MODEL_DIR / name).exists()]
if missing:
    raise RuntimeError(
        "Model files are missing. Run 'python train_models.py' first. Missing: "
        + ", ".join(missing)
    )

regression_model = joblib.load(MODEL_DIR / "regression_model.joblib")
classification_model = joblib.load(MODEL_DIR / "classification_model.joblib")
iris_target_names = joblib.load(MODEL_DIR / "iris_target_names.joblib")

app = FastAPI(title="Full-Stack Machine Learning App")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RegressionInput(BaseModel):
    # Eight example input fields are not used here: the diabetes dataset has 10 features.
    age: float = Field(..., description="Standardized age feature")
    sex: float = Field(..., description="Standardized sex feature")
    bmi: float = Field(..., description="Standardized BMI feature")
    bp: float = Field(..., description="Standardized blood pressure feature")
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class ClassificationInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    values = [[
        data.age, data.sex, data.bmi, data.bp, data.s1,
        data.s2, data.s3, data.s4, data.s5, data.s6
    ]]
    prediction = float(regression_model.predict(np.array(values))[0])
    return {"prediction": round(prediction, 2)}


@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    values = [[
        data.sepal_length, data.sepal_width,
        data.petal_length, data.petal_width
    ]]
    class_id = int(classification_model.predict(np.array(values))[0])
    return {
        "class_id": class_id,
        "prediction": str(iris_target_names[class_id])
    }
