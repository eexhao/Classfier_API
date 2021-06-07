from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import uvicorn
from fastapi import FastAPI, File, UploadFile

model = tf.keras.applications.InceptionV3(weights="imagenet")
input_size=(299,299)

def Image_process(file):
    image = Image.open(BytesIO(file))
    image = image.resize(input_size)
    image = np.asarray(image)[..., :3]
    image = image / 255.0
    image = np.expand_dims(image, axis =0)
    return image

def Classfication(image):
    predicts = model.predict(image)
    result = decode_predictions(predicts)[0]

    All_Results= []
    for i,item in enumerate(result):
        result_item = {}
        result_item["Object Class"] = item[1]
        result_item["Probability"] = item[2]
        #result_item["Probability"] = f"{item[2]*100:0.1f} %"
        All_Results.append(result_item)

    All_Results = sorted(All_Results, key=lambda k: k['Probability'], reverse=True)
    return All_Results
    # Final_Results = []
    # if All_Results[0].get('Probability')>0.5:
    #    Final_Results.append(All_Results[0])
    #
    # return Final_Results

app_desc = """<h2>Upload an image which contains a single common subject to`Image_Classifier`below:</h2>"""
app = FastAPI(title='Streamba challenge-Image Classifier', description=app_desc)

@app.post("/Image_Classifier")
async def Image_Classifier(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = Image_process(await file.read())
    prediction = Classfication(image)

    return prediction

if __name__ == "__main__":
    uvicorn.run(app, debug=True)