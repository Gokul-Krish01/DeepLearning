from fastapi import FastAPI,File , UploadFile
import numpy as np
import cv2
from deepface import DeepFace
import os
import uvicorn,time

app=FastAPI()
models={}
os.environ["DEEPFACE_HOME"]="/home/krish"

@app.on_event("startup")
def load():
    global models
    models["facenet"] = DeepFace.build_model("Facenet512")


@app.post("/analyze")
async def analyze(file1:UploadFile ,file2:UploadFile):
    start=time.time()
    content1 =await file1.read()
    nparr1=np.frombuffer(content1,np.uint8)
    img1 =cv2.imdecode(nparr1,cv2.IMREAD_COLOR)
   
    content2 = await file2.read()
    nparr2=np.frombuffer(content2,np.uint8)
    img2=cv2.imdecode(nparr2,cv2.IMREAD_COLOR)
    result=DeepFace.verify(img1_path=img1,img2_path=img2,model_name="Facenet512",detector_backend="retinaface",enforce_detection=False)
    end =time.time()
    total=end-start 
    if result['verified']:
        face='matched'
    else :
        face='unmatched'
    return {"status": "success", "data": face,"time":total}

@app.post("/count")
async def count(file1:UploadFile):
    st=time.time()
    content1=await file1.read()
    nparr1=np.frombuffer(content1,np.uint8)
    img1=cv2.imdecode(nparr1,cv2.IMREAD_COLOR)
    count=DeepFace.extract_faces(img_path=img1,detector_backend='retinaface',enforce_detection=False)
    end=time.time()
    total= end - st
    return {
        "status":"success",
        "data":len(count),
        "time":total
    }

if __name__=='__main__':
    uvicorn.run(app )
