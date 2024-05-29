import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Model 
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.inception_v3 import InceptionV3
from flask_cors import CORS



from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename


#função para classificar a imagem com a rede neural treinada previamente
def chestScanPrediction(path_file, _model):
    classes_dir = ["Adenocarcinoma", "Large cell carcinoma", "Normal", "Squamous cell carcinoma"]

    img = image.load_img(path_file, target_size=(400, 400))

    # Normalizing Image
    norm_img = image.img_to_array(img) / 255

    # Converting Image to Numpy Array
    input_arr_img = np.array([norm_img])

    # Getting Predictions
    pred = np.argmax(_model.predict(input_arr_img))

    # Printing Model Prediction
    print("Predicted Label:", classes_dir[pred])

    return classes_dir[pred]



# Chama a função de detecção de câncer e obtém os resultados
model_eff = load_model("./backend/vgg16_best_model.keras")
resultado = chestScanPrediction("./backend/uploads/8_-_Copy_3.png", model_eff)