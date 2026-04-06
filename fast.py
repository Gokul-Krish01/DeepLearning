from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import onnxruntime as ort
import numpy as np
app = FastAPI()

scale=joblib.load("scaler.joblib")
onxmodel=ort.InferenceSession("iris.onnx")
target = ["setosa", "versicolor", "virginica"]
class Features(BaseModel):
    feature1:int 
    feature2:int 
    feature3:int
    feature4:int


@app.post("/predict")
def predict(data:Features):    
    raw=np.array([data.features])
    scaled =scale.transform(raw).astype(np.float32)
    inputshape=onxmodel.get_inputs()[0].name
    out=onxmodel.run(None,{inputshape:scaled})
    out1=int(np.argmax(out[0]))

    return{
        "status": "success",
        "data": target[out1]
    }    
   
@app.get("/result")
def result(out1):
    
    return out1

# class Student(BaseModel):
#     name:str
#     age:int
#     dept:Optional["str"]=None

# @app.get("/")
# def home():
#     return {"message":"hii"}

# @app.post("/Student")
# def names(s: Student):
#     if s.dept:
#         return {"message": f"Welcome {s.name} from {s.dept}"}
#     return {"message": f"Welcome {s.name}"}