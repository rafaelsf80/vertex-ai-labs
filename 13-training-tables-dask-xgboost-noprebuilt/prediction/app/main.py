from fastapi.logger import logger
from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

import numpy as np
import os
import xgboost
import logging

# class ValidationErrorLoggingRoute(APIRoute):
#     def get_route_handler(self) -> Callable:
#         original_route_handler = super().get_route_handler()

#         async def custom_route_handler(request: Request) -> Response:
#             try:
#                 return await original_route_handler(request)
#             except RequestValidationError as exc:
#                 body = await request.body()
#                 detail = {"errors": exc.errors(), "body": body.decode()}
#                 raise HTTPException(status_code=422, detail=detail)

#         return custom_route_handler

import xgboost as xgb

app = FastAPI()
# app.router.route_class = ValidationErrorLoggingRoute

gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers

if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.INFO)


logger.info("Loading model")
print("loading model")

model_f = "/model/model.bst"
_model = xgb.Booster()
_model.load_model(model_f)


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    """ health check to ensure HTTP server is ready to handle 
        prediction requests
    """
    return {"status": "healthy"}


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()
    instances = body["instances"]

    inputs = np.asarray(instances)

    # Creamos el DMatrix
    dtest = xgb.DMatrix(inputs)
    
    # TRUCO: Forzamos a que el DMatrix de predicción use los mismos 
    # nombres que el modelo tenía durante el entrenamiento.
    # Esto elimina el error "training data did not have the following fields".
    dtest.feature_names = _model.feature_names
    
    outputs = _model.predict(dtest)

    return {"predictions": outputs.tolist()}