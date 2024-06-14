import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import requests
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
import tensorflow as tf
import mahotas as mt
from tensorflow.keras.models import load_model

class CancerDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cancer Detection App")


        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.zoom_factor = 1.0

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        process_menu = tk.Menu(menubar, tearoff=0)
        process_menu.add_command(label="Convert to Grayscale", command=self.converterCinza)
        process_menu.add_command(label="Generate Gray Histogram", command=self.generate_gray_histogram)
        process_menu.add_command(label="Generate 16-Tone Gray Histogram", command=self.generate_16_tone_gray_histogram)
        process_menu.add_command(label="Generate HSV Histogram", command=self.generate_hsv_histogram)
        process_menu.add_command(label="Generate HSV Histogram (16x8)", command=self.generate_hsv_histogram_16_8)
        process_menu.add_command(label="Calculate Co-occurrence Matrices", command=self.calculate_cooccurrence_matrices)
        process_menu.add_command(label="Haralick Descriptors", command=self.extract_haralick_descriptors)

        process_menu.add_command(label="Calculate Haralick Descriptors Matrices", command=self.calculate_haralick_descriptors_for_matrices)
        process_menu.add_command(label="Hu Moments Color", command=self.calculate_color_hu_moments)
        process_menu.add_command(label="Calculate Hu Moments", command=self.calculate_hu_moments)
        process_menu.add_command(label="Classify Sub-image", command=self.classify_sub_image)
        menubar.add_cascade(label="Process", menu=process_menu)

        zoom_menu = tk.Menu(menubar, tearoff=0)
        zoom_menu.add_command(label="Zoom In", command=self.zoom_in)
        zoom_menu.add_command(label="Zoom Out", command=self.zoom_out)
        menubar.add_cascade(label="Zoom", menu=zoom_menu)

        color_menu = tk.Menu(menubar, tearoff=0)
        color_menu.add_command(label="Convert to Color", command=self.convert_to_color)
        menubar.add_cascade(label="Color", menu=color_menu)

        self.image = None
        self.color_image = None  # Adicionando a inicialização do atributo color_image


        self.model = tf.keras.models.load_model("modelo_resnet50.h5", compile=False)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])




    def open_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.update_image()

    def update_image(self):
        self.image = cv2.imread(self.image_path)
        self.display_image()

    def display_image(self):
        if self.image is not None:
            zoomed_image = cv2.resize(self.image, None, fx=self.zoom_factor, fy=self.zoom_factor)
            image_rgb = cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)
            self.image_label.config(image=image_tk)
            self.image_label.image = image_tk

    def converterCinza(self):
        if hasattr(self, 'image'):
            self.color_image = self.image.copy()  # Salva uma cópia da imagem original
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = gray_image
            self.display_image()

    def convert_to_gray(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converter para tons de cinza
        resized_gray_image = cv2.resize(gray_image, (32, 32))  # Redimensionar para o tamanho desejado
        resized_gray_image = np.expand_dims(resized_gray_image, axis=-1)  # Adicionar a dimensão do canal
        return resized_gray_image 
    def convert_to_color(self):
        if hasattr(self, 'color_image'):  # Verifica se já existe uma versão colorida da imagem
            self.image = self.color_image  # Restaura a imagem colorida
            self.display_image()
        else:
            messagebox.showinfo("Info", "No grayscale image available")

    def generate_gray_histogram(self):
        if hasattr(self, 'image'):
            if len(self.image.shape) == 3:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = self.image
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            plt.plot(hist)
            plt.title('Gray Histogram')
            plt.xlabel('Gray Level')
            plt.ylabel('Frequency')
            plt.show()


    def generate_16_tone_gray_histogram(self):
        if hasattr(self, 'image'):
            if len(self.image.shape) == 3:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = self.image
            hist = cv2.calcHist([gray_image], [0], None, [16], [0, 256])
            plt.bar(range(16), hist.flatten(), width=0.8, align='center')
            plt.xticks(range(16))
            plt.title('16-Tone Gray Histogram')
            plt.xlabel('Gray Level')
            plt.ylabel('Frequency')
            plt.show()

    def generate_hsv_histogram(self):
        if hasattr(self, 'image'):
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

            plt.subplot(3, 1, 1)
            plt.plot(h_hist)
            plt.title('Hue Histogram')
            plt.xlabel('Hue')
            plt.ylabel('Frequency')

            plt.subplot(3, 1, 2)
            plt.plot(s_hist)
            plt.title('Saturation Histogram')
            plt.xlabel('Saturation')
            plt.ylabel('Frequency')

            plt.subplot(3, 1, 3)
            plt.plot(v_hist)
            plt.title('Value Histogram')
            plt.xlabel('Value')
            plt.ylabel('Frequency')

            plt.tight_layout()
            plt.show()

    def generate_hsv_histogram_16_8(self):
        if hasattr(self, 'image'):
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            h_bins = 16
            v_bins = 8
            hist = cv2.calcHist([hsv_image], [0, 2], None, [h_bins, v_bins], [0, 180, 0, 256])
            plt.imshow(hist, interpolation='nearest')
            plt.title('2D HSV Histogram (H: 16 bins, V: 8 bins)')
            plt.xlabel('Hue')
            plt.ylabel('Value')
            plt.colorbar()
            plt.show()

    # def calculate_cooccurrence_matrices(self):
    #     if hasattr(self, 'image'):
    #         gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    #         gray_image = (gray_image // 16).astype(np.uint8)
    #         distances = [1, 2, 4, 8, 16, 32]

    #         def cooccurrence_matrix(image, distance):
    #             max_gray = 16
    #             matrix = np.zeros((max_gray, max_gray), dtype=np.float32)
    #             rows, cols = image.shape
    #             for row in range(rows):
    #                 for col in range(cols):
    #                     if row + distance < rows and col + distance < cols:
    #                         current_pixel = image[row, col]
    #                         right_pixel = image[row, col + distance]
    #                         bottom_pixel = image[row + distance, col]
    #                         bottom_right_pixel = image[row + distance, col + distance]
    #                         matrix[current_pixel, right_pixel] += 1
    #                         matrix[current_pixel, bottom_pixel] += 1
    #                         matrix[current_pixel, bottom_right_pixel] += 1

    #             matrix /= np.sum(matrix)
    #             return matrix

    #         cooccurrence_matrices = {distance: cooccurrence_matrix(gray_image, distance) for distance in distances}

    #         fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    #         axes = axes.flatten()
    #         for ax, distance in zip(axes, distances):
    #             ax.imshow(cooccurrence_matrices[distance], cmap='gray')
    #             ax.set_title(f'Distance: {distance}')
    #             ax.axis('off')
    #         plt.tight_layout()
    #         plt.show()
    
    def calculate_cooccurrence_matrices(self):
        if hasattr(self, 'image'):
            # Convertendo a imagem para tons de cinza
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # Quantizando a imagem para 16 tons de cinza
            gray_image = (gray_image // 16).astype(np.uint8)
            # Definindo as distâncias
            distances = [1, 2, 4, 8, 16, 32]

            def cooccurrence_matrix(image, distance):
                max_gray = 16
                matrix = np.zeros((max_gray, max_gray), dtype=np.float32)
                rows, cols = image.shape
                for row in range(rows):
                    for col in range(cols):
                        if row + distance < rows and col + distance < cols:
                            current_pixel = image[row, col]
                            right_pixel = image[row, col + distance]
                            bottom_pixel = image[row + distance, col]
                            bottom_right_pixel = image[row + distance, col + distance]
                            matrix[current_pixel, right_pixel] += 1
                            matrix[current_pixel, bottom_pixel] += 1
                            matrix[current_pixel, bottom_right_pixel] += 1
                return matrix

            # Criando uma string para armazenar os resultados
            result_string = ""
            # Calculando e exibindo as matrizes de co-ocorrência
            for distance in distances:
                co_matrix = cooccurrence_matrix(gray_image, distance)
                # Convertendo os valores da matriz para inteiros
                co_matrix_int = co_matrix.astype(int)
                # Formatando a matriz como uma string
                matrix_str = "\n".join(["\t".join(map(str, row)) for row in co_matrix_int])
                result_string += f"Co-occurrence Matrix for Distance {distance}:\n{matrix_str}\n\n"

            # Criando uma nova janela para exibir o resultado com rolagem
            scroll_window = tk.Toplevel(self.root)
            scroll_window.title("Co-occurrence Matrices")

            text_widget = tk.Text(scroll_window, wrap='none')
            text_widget.insert(tk.END, result_string)
            text_widget.config(state=tk.DISABLED)

            x_scroll = tk.Scrollbar(scroll_window, orient='horizontal', command=text_widget.xview)
            y_scroll = tk.Scrollbar(scroll_window, orient='vertical', command=text_widget.yview)
            text_widget.config(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
            y_scroll.pack(side=tk.RIGHT, fill=tk.Y)


    def extract_haralick_descriptors(self):
        if hasattr(self, 'image'):
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            textures = mt.features.haralick(gray_image)
            mean_desc = textures.mean(axis=0)
            messagebox.showinfo("Haralick Descriptors", 
                                f"Contrast: {mean_desc[0]}\n"
                                f"Dissimilarity: {mean_desc[1]}\n"
                                f"Homogeneity: {mean_desc[2]}\n"
                                f"ASM: {mean_desc[3]}\n"
                                f"Energy: {mean_desc[4]}\n"
                                f"Correlation: {mean_desc[5]}")
    def calculate_color_hu_moments(self):
        if self.color_image is not None:
            # Splitting the image into color channels
            b, g, r = cv2.split(self.color_image)

            # Calculating Hu moments for each channel
            channels_hu_moments = []
            for channel in [b, g, r]:
                moments = cv2.moments(channel)
                huMoments = cv2.HuMoments(moments)
                huMoments = [-1 * np.sign(hu) * np.log10(np.abs(hu)) if hu != 0 else 0 for hu in huMoments]
                channels_hu_moments.append(huMoments)

            messagebox.showinfo("Hu Moments (Color)", f"Hu Moments (Blue Channel):\n{channels_hu_moments[0]}\n\n"
                                                       f"Hu Moments (Green Channel):\n{channels_hu_moments[1]}\n\n"
                                                       f"Hu Moments (Red Channel):\n{channels_hu_moments[2]}")

    def calculate_haralick_descriptors_for_matrices(self):
        if hasattr(self, 'image'):
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            gray_image = (gray_image // 16).astype(np.uint8)
            distances = [1, 2, 4, 8, 16, 32]

            def cooccurrence_matrix(image, distance):
                max_gray = 16
                matrix = np.zeros((max_gray, max_gray), dtype=np.float32)
                rows, cols = image.shape
                for row in range(rows):
                    for col in range(cols):
                        if row + distance < rows and col + distance < cols:
                            current_pixel = image[row, col]
                            right_pixel = image[row, col + distance]
                            bottom_pixel = image[row + distance, col]
                            bottom_right_pixel = image[row + distance, col + distance]
                            matrix[current_pixel, right_pixel] += 1
                            matrix[current_pixel, bottom_pixel] += 1
                            matrix[current_pixel, bottom_right_pixel] += 1
                return matrix

            def calculate_haralick_features(matrix):
                total = np.sum(matrix)
                norm_matrix = matrix / total

                # Contraste
                contrast = 0
                for i in range(norm_matrix.shape[0]):
                    for j in range(norm_matrix.shape[1]):
                        contrast += (i - j) ** 2 * norm_matrix[i, j]

                # Homogeneidade
                homogeneity = 0
                for i in range(norm_matrix.shape[0]):
                    for j in range(norm_matrix.shape[1]):
                        homogeneity += norm_matrix[i, j] / (1 + abs(i - j))

                # Entropia
                entropy = 0
                for i in range(norm_matrix.shape[0]):
                    for j in range(norm_matrix.shape[1]):
                        if norm_matrix[i, j] > 0:
                            entropy -= norm_matrix[i, j] * np.log2(norm_matrix[i, j])

                return contrast, homogeneity, entropy

            result_string = ""
            for distance in distances:
                co_matrix = cooccurrence_matrix(gray_image, distance)
                contrast, homogeneity, entropy = calculate_haralick_features(co_matrix)
                result_string += (f"Distance {distance}:\n"
                                  f"Contrast: {contrast:.4f}\n"
                                  f"Homogeneity: {homogeneity:.4f}\n"
                                  f"Entropy: {entropy:.4f}\n\n")

            scroll_window = tk.Toplevel(self.root)
            scroll_window.title("Haralick Descriptors")

            text_widget = tk.Text(scroll_window, wrap='none')
            text_widget.insert(tk.END, result_string)
            text_widget.config(state=tk.DISABLED)

            x_scroll = tk.Scrollbar(scroll_window, orient='horizontal', command=text_widget.xview)
            y_scroll = tk.Scrollbar(scroll_window, orient='vertical', command=text_widget.yview)
            text_widget.config(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
            y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    def calculate_hu_moments(self):
        if hasattr(self, 'image'):
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            moments = cv2.moments(gray_image)
            hu_moments = cv2.HuMoments(moments).flatten()
            messagebox.showinfo("Hu Moments", "\n".join([f"Hu Moment {i+1}: {hu_moments[i]}" for i in range(7)]))

    def classify_sub_image(self, event=None):
        if self.image is None:
            messagebox.showerror("Erro", "Nenhuma imagem carregada.")
            return

        # Convertendo a imagem para tons de cinza
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Convertendo a imagem em tons de cinza para RGB
        rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

        # Redimensionando a imagem para o tamanho esperado pelo modelo
        sub_image_resized = cv2.resize(rgb_image, (100, 100))

        # Normalizando os valores de pixel para o intervalo [0, 1]
        sub_image_resized = sub_image_resized / 255.0

        # Adicionando uma dimensão para compatibilidade com o modelo
        sub_image_resized = np.expand_dims(sub_image_resized, axis=0)

        #  Realizando a previsão usando o modelo
        predictions = self.model.predict(sub_image_resized)

        # Obtendo o índice da classe com maior probabilidade
        class_index = np.argmax(predictions)

        # Lista de rótulos de classe correspondentes às classes esperadas pelo modelo
        class_labels = ["ASC-H", "ASC-US", "HSIL", "LSIL", "Negative for intraepithelial lesion", "SCC"]

        # Verificando se o índice está dentro do intervalo válido
        if 0 <= class_index < len(class_labels):
            classification_result = class_labels[class_index]
        else:
            classification_result = "Classe desconhecida"

        # Exibindo o resultado da classificação
        messagebox.showinfo("Resultado da Classificação", f"Classe Predita: {classification_result}")
        self.clear_image()

    def clear_image(self):
        self.image = None
        self.color_image = None
        # Você pode adicionar qualquer outra limpeza ou redefinição necessária aqui
    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.display_image()

    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.display_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = CancerDetectionApp(root)
    root.mainloop()
