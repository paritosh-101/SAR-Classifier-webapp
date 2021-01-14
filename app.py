# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:53:50 2021

@author: user
"""

import os
import tensorflow as tf
import numpy as np
from keras.models import load_model

from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import ReLU
from keras.layers import Activation
from keras.layers import BatchNormalization
from os import listdir
from numpy import asarray
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from flask import Flask, request
from flask import render_template
# from keras.backend import set_session

app = Flask(__name__)

UPLOAD_FOLDER = 'D:\\Paritosh\\_workspace\\webapp\\static'

global model
model = load_model("c_model_043080.h5")

# @app.route('/')
# def welc_msg():
#     return "Stairway to heaven..."

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
                )
            image_file.save(image_location)
            img = load_img(os.path.join(UPLOAD_FOLDER, image_file.filename))
            image_arr = img_to_array(img)
            image_arr = image_arr[:,:,0]/255.0
            image_arr = image_arr.reshape((1,256,256,1))
            pred = model.predict(image_arr)
            pred = np.around(pred, decimals=3)
            return render_template("index.html", prediction=pred)
    return render_template("index.html", prediction=0)





if __name__ == '__main__':
    app.run(host='0.0.0.0')