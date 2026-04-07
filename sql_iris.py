from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column , Integer,String, Float,DateTime
from sqlalchemy.sql import func
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import onnxruntime as ort
import numpy as np
import time
app = FastAPI()


dburl="mysql+pymysql://root:krish001@localhost:3306/iris"

engine= create_engine(dburl)
session=sessionmaker(bind=engine)
base=declarative_base()

scale=joblib.load("scaler.joblib")
onxmodel=ort.InferenceSession("iris.onnx")
target = ["setosa", "versicolor", "virginica"]

class Features(BaseModel):
    feature1:float
    feature2:float
    feature3:float
    feature4:float

class Prediction(base):
    __tablename__ ="prediction"
    id= Column(Integer , primary_key=True,index=True)
    feature1=Column(Float)
    feature2=Column(Float)
    feature3=Column(Float)
    feature4=Column(Float)
    flower_name=Column(String(50))
    time =Column(DateTime(timezone=True),server_default=func.now())
    model_time=Column(Float)
base.metadata.create_all(bind=engine)

@app.post("/predict")
def predict(data:Features): 
    st=time.time()
    raw=np.array([[data.feature1,data.feature2,data.feature3,data.feature4]])
    scaled =scale.transform(raw).astype(np.float32)
    inputshape=onxmodel.get_inputs()[0].name
    out=onxmodel.run(None,{inputshape:scaled})
    out1=int(np.argmax(out[0]))
    name =target[out1]
    et=time.time()
    total=et-st
    db=session()
    try:
     new_data=Prediction(
     feature1=data.feature1,
     feature2=data.feature2,
     feature3=data.feature3,
     feature4=data.feature4,
     flower_name=name,
     model_time=total*1000
     )
     db.add(new_data)
     db.commit()
     db.refresh(new_data)

    finally:
     db.close()
    return{
        "status": "success",
        "data": name
    }  

@app.get("/history")
def history():
   db=session()
   data1=db.query(Prediction).all()
   db.close()

   return{
      "status":"success",
      "data":data1
   }


