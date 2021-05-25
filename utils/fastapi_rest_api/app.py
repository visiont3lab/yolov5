import sys
import os
import uvicorn
from fastapi import FastAPI, Request,File, UploadFile, status, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.logger import logger
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

import base64
import torchvision

# pip install fastapi uvicorn aiofiles
# openssl req -newkey rsa:2048 -nodes -keyout key.pem -x509 -days 365 -out certificate.pem

# Init Fast Api
app = FastAPI()
# CORS Support
origins = [
    'http://localhost:8000',
    'http://localhost:8081'
]
app.add_middleware(CORSMiddleware,
                    allow_origins=origins,
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                    )

templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
async def startup_event():
   # Setup 
    path2weights = "../datasets/speech2text/models/best_model_custom.pt"
    weights = torch.load(path2weights,map_location=torch.device('cpu'))
    
    global net

    net.load_state_dict(weights)    
    net.eval()

    global text_transform
    text_transform = TextTransform()


class JsonPredictRequest(BaseModel):
    message:  Optional[str] =  None
  
@app.post("/predict", response_class=JSONResponse)
def process(item: JsonPredictRequest):
    

if __name__ == "__main__":
    uvicorn.run(app, reload=False,
                host="0.0.0.0", port=8000),
                #ssl_keyfile="./cert/key.pem", 
                #ssl_certfile="./cert/certificate.pem")