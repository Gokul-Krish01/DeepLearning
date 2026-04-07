from fastapi import FastAPI
import onnxruntime as ort
import joblib
import numpy as np
from PIL import Image

app=FastAPI()

scale=joblib.load("Scale.joblib")
model=ort.InferenceSession("digitd.onnx")

@app.post('/predict')
def predict(img_path):
    img = Image.open(img_path).convert('L').resize((8, 8))
    img_array = np.array(img).astype(np.float32)

# This math is critical for the 8x8 digits model
    img_array = 16-(img_array * 16.0 / 255.0)

    features = img_array.flatten().reshape(1, -1)
    scaled_feature = scale.transform(features).astype(np.float32)
    scaled_feature=scale.transform(features)
    input_shape=model.get_inputs()[0].name
    output_shape=model.get_outputs()[0].name

    output=model.run([output_shape] ,{input_shape:scaled_feature})
    result = int(np.argmax(output[0]))

    return {
        "output":result,
        "status":"success"
    }