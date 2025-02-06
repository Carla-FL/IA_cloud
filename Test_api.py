
from flask import Flask, request
import numpy as np
from PIL import Image
import base64
import re
from io import BytesIO


# TensorFlow and tf.keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

model = MobileNetV2(weights='imagenet')
# Fonction de prédiction
def model_predict(img, model):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')
    preds = model.predict(x)
    return preds
    
def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)

    #Only for testing
    #with open('reco_output.jpg', "wb") as fh:
    #    fh.write(BytesIO(base64.decodebytes(bytes(((img_base64.split(','))[1]), "utf-8"))).read())

    #pil_image = Image.open('reco_output.jpg')

    pil_image = Image.open(BytesIO(base64.decodebytes(bytes(((img_base64.split(','))[1]), "utf-8"))))
    pil_image.convert('RGB')
    return pil_image

app = Flask(__name__)

@app.route('/')
def index():
    return "Bienvenue sur l'API Image Recognotion"

@app.route('/api/image/recognition/', methods=['POST'])
def predict():

    if request.form.get("image"):
        image = request.form.get("image")
        print(image)
        image_pillow = base64_to_pil(image)
        result = image_classification(image_pillow) # methode 2
        #result = image_classification(image) # methode 1
        return result
    else:
        return "Error"


def image_classification(img):
    # img = Image.open("./image.jpeg")
    # img = Image.open(image) # pas besoin pur le methode 2
    # prediction
    preds = model_predict(img, model)
    pred_proba = "{:.3f}".format(np.amax(preds))
    pred_class = decode_predictions(preds, top=1)
    result = str(pred_class[0][0][1])
    result = result.replace('_', ' ').capitalize()
    return ({"result":result, "probability":pred_proba})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)