import pandas as pd
import os
from PIL import Image

# Ler o arquivo classifications.csv
classifications = pd.read_csv('classifications.csv')

# Criar diretório para armazenar as sub-imagens
output_dir = 'sub_images'
os.makedirs(output_dir, exist_ok=True)

# Dicionário para contar quantas imagens foram salvas para cada categoria de Bethesda
contadores = {}

# Função para recortar e armazenar a sub-imagem
def recortar_e_armazenar(imagem_path, nucleus_x, nucleus_y, bethesda_system, cell_id):
    # Verificar se já salvamos 20 imagens dessa classificação
    if contadores.get(bethesda_system, 0) >= 20:
        return

    try:
        imagem = Image.open(imagem_path)
    except FileNotFoundError:
        return
    
    x = int(nucleus_x)
    y = int(nucleus_y)
    w = 100  # Largura da sub-imagem
    h = 100  # Altura da sub-imagem
    celula = imagem.crop((x, y, x+w, y+h))
    sub_dir = os.path.join(output_dir, bethesda_system)
    os.makedirs(sub_dir, exist_ok=True)
    sub_imagem_path = os.path.join(sub_dir, f'{cell_id}.png')
    celula.save(sub_imagem_path)

    # Atualizar contador
    contadores[bethesda_system] = contadores.get(bethesda_system, 0) + 1

# Iterar sobre as classificações e recortar as sub-imagens
for index, row in classifications.iterrows():
    image_filename = os.path.join('dataset', row['image_filename'])  
    nucleus_x = row['nucleus_x']
    nucleus_y = row['nucleus_y']
    bethesda_system = row['bethesda_system']
    cell_id = row['cell_id']
    recortar_e_armazenar(image_filename, nucleus_x, nucleus_y, bethesda_system, cell_id)
