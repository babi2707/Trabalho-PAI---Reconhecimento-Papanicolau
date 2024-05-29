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

app = Flask(__name__)

# Aplicar CORS à aplicação Flask
CORS(app)

# Configurações para upload de arquivos
UPLOAD_FOLDER = 'backend/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#função para classificar a imagem com a rede neural treinada previamente
def chestScanPrediction(path_file, _model):
    classes_dir = ["Câncer", "Câncer", "Normal", "Câncer"]

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

    

def limpar_pasta(pasta):
    # Percorre todos os arquivos na pasta
    for arquivo in os.listdir(pasta):
        # Cria o caminho completo do arquivo
        caminho_arquivo = os.path.join(pasta, arquivo)
        # Verifica se é um arquivo regular
        if os.path.isfile(caminho_arquivo):
            # Remove o arquivo
            os.remove(caminho_arquivo)

# Função para verificar se a extensão do arquivo é permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Rota para receber a imagem e retornar os resultados da detecção de câncer
@app.route('/detect-cancer', methods=['POST'])
def detect_cancer():
    # Verifica se o arquivo foi enviado na requisição
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']

    # Verifica se o arquivo é válido
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Salva o arquivo na pasta de uploads
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Chama a função de detecção de câncer e obtém os resultados
        model_eff = load_model("./backend/vgg16_best_model.keras")
        resultado = chestScanPrediction("./backend/uploads/" + filename, model_eff)
        limpar_pasta("./backend/uploads/")

        response = jsonify({'type': resultado})
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response

    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)