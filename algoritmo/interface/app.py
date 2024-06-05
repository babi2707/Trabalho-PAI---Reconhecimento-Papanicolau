from flask import Flask, request, jsonify
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage.feature.texture import greycomatrix, greycoprops
from skimage.measure import moments_hu
from werkzeug.utils import secure_filename
import os
import skimage

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Função para salvar imagem
def save_image(file):
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath

# Função para converter para tons de cinza
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Função para gerar histograma
def generate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist.flatten()

# Função para gerar histograma HSV
def generate_hsv_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
    return {
        'h_hist': h_hist.tolist(),
        's_hist': s_hist.tolist(),
        'v_hist': v_hist.tolist()
    }

# Função para extrair descritores de Haralick
def extract_haralick(image):
    glcm = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast').mean()
    dissimilarity = greycoprops(glcm, 'dissimilarity').mean()
    homogeneity = greycoprops(glcm, 'homogeneity').mean()
    energy = greycoprops(glcm, 'energy').mean()
    correlation = greycoprops(glcm, 'correlation').mean()
    ASM = greycoprops(glcm, 'ASM').mean()
    return {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'ASM': ASM
    }

# Função para extrair momentos de Hu
def extract_hu_moments(image):
    moments = cv2.moments(image)
    huMoments = cv2.HuMoments(moments)
    # Escala logarítmica dos momentos de Hu
    for i in range(0,7):
        huMoments[i] = -1 * np.sign(huMoments[i]) * np.log10(np.abs(huMoments[i]))
    return huMoments.flatten()

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filepath = save_image(file)
    image = cv2.imread(filepath)

    gray_image = convert_to_gray(image)
    histogram = generate_histogram(gray_image)
    hsv_histogram = generate_hsv_histogram(image)
    haralick = extract_haralick(gray_image)
    hu_moments = extract_hu_moments(gray_image)

    return jsonify({
        'histogram': histogram.tolist(),
        'hsv_histogram': hsv_histogram,
        'haralick': haralick,
        'hu_moments': hu_moments.tolist()
    })

# Função para carregar o modelo de classificação (exemplo fictício)
def load_classification_model():
    # Aqui você carregaria seu modelo de classificação, como um modelo treinado com TensorFlow, PyTorch, etc.
    # Neste exemplo, vamos apenas retornar uma função de classificação aleatória.
    def classify_image(image):
        # Aqui você implementaria a lógica real para classificar a imagem
        # Neste exemplo fictício, estamos retornando uma classe aleatória
        classes = ['Normal', 'Cancer']
        return np.random.choice(classes)

    return classify_image

classify_image = load_classification_model()

@app.route('/classify', methods=['POST'])
def classify_sub_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_data = image_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Aqui você pode realizar pré-processamento na imagem, se necessário

    # Classificação da imagem
    prediction = classify_image(image)

    return jsonify({'class': prediction})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
