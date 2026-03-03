from fastapi import FastAPI
from pathlib import Path
from src import evaluate as utils_evaluate
from pydantic import BaseModel, ConfigDict
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

models = utils_evaluate.load_model(MODEL_PATH)

app = FastAPI()


class CreditRiskFeatures(BaseModel):
    fea_1: int
    fea_2: float
    fea_3: int
    fea_4: float
    fea_5: int
    fea_6: int
    fea_7: int
    fea_8: int
    fea_9: int
    fea_10: int
    fea_11: float
    OVD_t1: int
    OVD_t2: int
    OVD_t3: int
    OVD_sum: int
    pay_normal: int
    prod_limit: float
    new_balance: float
    highest_balance: float
    count: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fea_1": 4, "fea_2": 1415.5, "fea_3": 3, "fea_4": 550000.0,
                "fea_5": 1, "fea_6": 15, "fea_7": 5, "fea_8": 100,
                "fea_9": 4, "fea_10": 60000, "fea_11": 1.2,
                "OVD_t1": 0, "OVD_t2": 0, "OVD_t3": 0, "OVD_sum": 0,
                "pay_normal": 12, "prod_limit": 50000.0, "new_balance": 1250.5,
                "highest_balance": 3500.0, "count": 1
            }
        }
    )


@app.post("/predict")
def predict(data: CreditRiskFeatures):
    df = pd.DataFrame([data.model_dump()])

    y_pred = models.predict(df)
    prediction_finale = int(y_pred[0])

    return {"class": prediction_finale}
