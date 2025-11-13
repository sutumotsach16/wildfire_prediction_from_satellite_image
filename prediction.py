import tkinter as tk #https://fptshop.com.vn/tin-tuc/danh-gia/tkinter-python-159345
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np #https://www.w3schools.com/python/numpy/numpy_intro.asp
import tensorflow as tf

# Giả sử đã có mô hình nhận diện cháy rừng đã huấn luyện và lưu tại wildfire_detection_model.h5
MODEL_PATH = r'D:\project\wildfire_prediction_dataset\v1\wildfire_detection_model.keras'

# Hàm dự đoán cháy rừng
def predict_fire(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    img = Image.open(image_path).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction[0][0] > 0.5  # True nếu có cháy rừng

# Giao diện
class WildfireApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện cháy rừng từ ảnh vệ tinh")
        self.img_label = tk.Label(root)
        self.img_label.pack()
        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack()
        tk.Button(root, text="Chọn ảnh", command=self.load_image).pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            img = Image.open(file_path).resize((256, 256))
            self.img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.img)
            try:
                has_fire = predict_fire(file_path)
                if has_fire:
                    self.result_label.config(text="Phát hiện cháy rừng!", fg="red")
                else:
                    self.result_label.config(text="Không phát hiện cháy rừng.", fg="green")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi dự đoán: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = WildfireApp(root)
    root.mainloop()