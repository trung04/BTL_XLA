import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import os


class ShapeRecognizer:
    def __init__(self, model_path="shape_cnn_model_color.h5", target_size=(64, 64)):
        """
        Khởi tạo đối tượng và tải model CNN.
        """
        self.model_path = model_path
        self.target_size = target_size

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình: {model_path}")

        print(f"📦 Đang tải mô hình từ: {model_path}")
        self.model = load_model(model_path)
        print("✅ Mô hình đã tải thành công!")

    # ==========================
    # 1️⃣ Đọc ảnh RGB thủ công
    # ==========================
    def load_image(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        print(f"Ảnh gốc: {img.shape}")
        return img

    # ==========================
    # 2️⃣ Chuyển RGB → Grayscale thủ công
    # ==========================
    def rgb_to_gray(self, img):
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        return np.clip(gray, 0, 255).astype(np.uint8)

    # ==========================
    # 3️⃣ Resize thủ công
    # ==========================
    def resize(self, img, new_w=None, new_h=None):
        if new_w is None or new_h is None:
            new_w, new_h = self.target_size

        h, w = img.shape[:2]
        if len(img.shape) == 3:
            c = img.shape[2]
            out = np.zeros((new_h, new_w, c), dtype=np.uint8)
            for y in range(new_h):
                for x in range(new_w):
                    src_x = int(x * w / new_w)
                    src_y = int(y * h / new_h)
                    out[y, x] = img[src_y, src_x]
        else:
            out = np.zeros((new_h, new_w), dtype=np.uint8)
            for y in range(new_h):
                for x in range(new_w):
                    src_x = int(x * w / new_w)
                    src_y = int(y * h / new_h)
                    out[y, x] = img[src_y, src_x]
        return out

    # ==========================
    # 4️⃣ Chuẩn hóa cho model
    # ==========================
    def normalize(self, img):
        img = img / 255.0
        img = np.expand_dims(img, axis=0)  # (1,64,64,3)
        return img

    # ==========================
    # 5️⃣ Pipeline xử lý ảnh
    # ==========================
    def preprocess(self, path, show=False):
        """
        Đọc, resize, hiển thị (tùy chọn), rồi chuẩn hóa.
        """
        img = self.load_image(path)
        resized = self.resize(img, *self.target_size)

        if show:
            plt.figure(figsize=(8, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Ảnh gốc")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(resized)
            plt.title("Ảnh sau resize 64x64")
            plt.axis("off")
            plt.show()

        return self.normalize(resized)

    # ==========================
    # 6️⃣ Dự đoán hình học
    # ==========================
    def predict(self, path, show=True):
        x = self.preprocess(path)
        pred = self.model.predict(x)[0]
        label = ["circle", "rectangle"][np.argmax(pred)]
        confidence = float(np.max(pred))
        return label, confidence