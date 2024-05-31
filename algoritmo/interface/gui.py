import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
import requests
import json

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
        process_menu.add_command(label="Convert to Grayscale", command=self.convert_to_gray)
        process_menu.add_command(label="Generate Gray Histogram", command=self.generate_gray_histogram)
        process_menu.add_command(label="Generate HSV Histogram", command=self.generate_hsv_histogram)
        process_menu.add_command(label="Haralick Descriptors", command=self.extract_haralick_descriptors)
        process_menu.add_command(label="Hu Moments", command=self.extract_hu_moments)
        process_menu.add_command(label="Classify Sub-image", command=self.classify_sub_image)  
        menubar.add_cascade(label="Process", menu=process_menu)

        zoom_menu = tk.Menu(menubar, tearoff=0)
        zoom_menu.add_command(label="Zoom In", command=self.zoom_in)
        zoom_menu.add_command(label="Zoom Out", command=self.zoom_out)
        menubar.add_cascade(label="Zoom", menu=zoom_menu)

        color_menu = tk.Menu(menubar, tearoff=0)
        color_menu.add_command(label="Convert to Color", command=self.convert_to_color)
        menubar.add_cascade(label="Color", menu=color_menu)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg")])
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image()

    def display_image(self):
        zoomed_image = cv2.resize(self.image, None, fx=self.zoom_factor, fy=self.zoom_factor)
        image_rgb = cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

    def convert_to_gray(self):
        if hasattr(self, 'image'):
            self.color_image = self.image.copy()  # Salva uma cópia da imagem original
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = gray_image
            self.display_image()


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

    def extract_haralick_descriptors(self):
        if hasattr(self, 'image'):
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, encoded_image = cv2.imencode('.png', gray_image)
            files = {'image': ('image.png', encoded_image.tobytes(), 'image/png')}
            response = requests.post('http://localhost:5000/upload', files=files)
            if response.status_code == 200:
                data = response.json()
                messagebox.showinfo("Haralick Descriptors", f"Contrast: {data['haralick']['contrast']}\nDissimilarity: {data['haralick']['dissimilarity']}\nHomogeneity: {data['haralick']['homogeneity']}\nEnergy: {data['haralick']['energy']}\nCorrelation: {data['haralick']['correlation']}\nASM: {data['haralick']['ASM']}")
            else:
                messagebox.showerror("Error", "Failed to extract Haralick descriptors.")
            

    def extract_hu_moments(self):
        if hasattr(self, 'image'):
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            moments = cv2.moments(gray_image)
            huMoments = cv2.HuMoments(moments)
            # Log scale hu moments
            for i in range(0,7):
                huMoments[i] = -1 * np.sign(huMoments[i]) * np.log10(np.abs(huMoments[i]))
            messagebox.showinfo("Hu Moments", f"Hu Moments:\n{huMoments.flatten()}")
            

    def classify_sub_image(self):  # Added this method
        if hasattr(self, 'image'):
            sub_image = self.image[100:300, 100:300]  # Example: select a sub-image
            _, encoded_image = cv2.imencode('.png', sub_image)
            files = {'image': ('image.png', encoded_image.tobytes(), 'image/png')}
            response = requests.post('http://localhost:5000/classify', files=files)
            if response.status_code == 200:
                data = response.json()
                predicted_class = data.get('class', 'Unknown')
                messagebox.showinfo("Sub-image Classification", f"The predicted class is: {predicted_class}")
            else:
                messagebox.showerror("Error", "Failed to classify the sub-image")

    def zoom_in(self):
        self.zoom_factor *= 1.1
        self.display_image()

    def zoom_out(self):
        self.zoom_factor /= 1.1
        self.display_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = CancerDetectionApp(root)
    root.mainloop()

