from fastapi import FastAPI
from predictor import get_models, find_cuisine, find_closest

app = FastAPI()

# @app.get("/")
# def read_root():
#     svc, nn = get_models()

#     return {"Hello": "World"}

@app.post("/predict")
def predict(ingredients: list[str], N: int = 5):
    output = {}

    svc, nn = get_models()
    output["cuisine"], output["score"] = find_cuisine(svc, ingredients)
    output["closest"] = find_closest(nn, ingredients, N)

    return output