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
        result_item["Probability"] = float("{:.1f}".format(item[2]*100))
        All_Results.append(result_item)

    if All_Results[0].get('Probability') > 50:
        Final_Result='This is a {}, and it is {}% correct'.format(All_Results[0].get('Object Class'),All_Results[0].get('Probability'))
    else:
        Final_Result = 'This is likely to be a {} or {}, and the probabilities are {}% and {}% respectively'.format(
            All_Results[0].get('Object Class'), All_Results[1].get('Object Class'),
            All_Results[0].get('Probability'), All_Results[1].get('Probability'))
    return Final_Result

app_desc = """<h2>Upload an image which contains a single common subject to`Image_Classifier`below:</h2>"""
app = FastAPI(title='Streamba challenge-Image Classifier', description=app_desc)

@app.post("/Image_Classifier")
async def Image_Classifier(file: UploadFile = File(...)):
    image = Image_process(await file.read())
    prediction = Classfication(image)
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, debug=True)