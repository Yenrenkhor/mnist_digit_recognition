from flask import Flask, render_template, request
import os
import sys
import re
import numpy as np
from scipy.misc import imread, imresize
import base64
from digit_web.model.load_model import *

app = Flask(__name__)

# gloabal vars for easy reusability
global model, graph

# initialize variables
model, graph = init()

# Decoding image from base64 into raw representation
def convertImage(imgData):
    img_str = re.search(r'base64,(.*)', str(imgData)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(img_str))


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    '''
    Whenever predict is called, we upload user input
    as image into the model
    :return: the prediction from our model
    '''

    # get raw data format of the image
    imgData = request.get_data()

    # encode it into a suitable format
    convertImage(imgData)

    # read the image into memory
    x = imread('output.png', mode='L')
    # change image size
    x = imresize(x, (28, 28))
    # convert to a 4D tensor to feed into model
    x = x.reshape(1, 28, 28, 1)
    x = x.astype(float)

    out = model.predict(x)
    print(np.argmax(out, axis=1))
    # Convert the response to string
    response = np.argmax(out, axis=1)
    return str(response[0])

    # with graph.as_default():
    #     # perform prediction
    #     out = model.predict(x)
    #     print(np.argmax(out, axis=1))
    #     # Convert the response to string
    #     response = np.argmax(out, axis=1)
    #     return str(response[0])



if __name__ == '__main__':
    app.run()
