import pickle
import uvicorn, logging, time, os
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.routing import APIRoute
from typing import Callable,  Tuple
import numpy as np

from logging.config import dictConfig

class LogConfig(BaseModel):
    """Logging configuration to be set for the server"""

    LOGGER_NAME: str = "sd_opt_api"
    LOG_FORMAT: str = "%(levelprefix)s %(asctime)s  %(message)s"
    LOG_LEVEL: str = "DEBUG"

    # Logging config
    version = 1
    disable_existing_loggers = False
    formatters = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    }
    loggers = {
        LOGGER_NAME: {"handlers": ["default"], "level": LOG_LEVEL},
    }


dictConfig(LogConfig().dict())
logger = logging.getLogger(LogConfig().LOGGER_NAME)


class CustomAuthHeaders(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:

            id_header = ("x-request-id".encode(), f"hi".encode(), )
            request.headers.__dict__["_list"].append(id_header)

            groups = request.headers.get("x-test-groups")
            if groups:
                groups = groups.split(",")
                logger.info(f"GROUPS: {groups}")
            else:
                logger.info(f"NO GROUPS: {groups}")    
            return await original_route_handler(request)

        return custom_route_handler

router = APIRouter(route_class=CustomAuthHeaders)

app = FastAPI(debug=os.getenv("DEBUG", False),)


@app.get("/")
async def root(request: Request):
    logger.info("LOGGG")
    logger.debug("LOGGG")
    headers = request.headers
    return {
        "username": f"{headers.get('x-auth-request-email-custom-parsed')}",
        "groups": f"{headers.get('x-auth-request-groups-custom-parsed')}",
        "time":  f"{headers.get('x-auth-request-process-time')}",
        }

@app.get("/test")
@router.get("/router/test")
def test_run(request: Request):
    logger.info(f"METHOD: {request.method}")
    return {"method": request.method} if  request.method == "debug" else request.headers

@app.get("/predict/{model}")
def predict(request: Request,model) -> dict:
    model = pickle.load(open(f"models/{model}.pkl", 'rb'))
    data = {"Pclass": 3, "Sex": 1, "Age": 22, "SibSp": 1, "Parch": 2, "Fare": 30.3, "Embarked":2}
    predict = model.predict([list(data.values())])[0]
    return {"predict": f"{predict}"} 

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        proxy_headers=True
    )