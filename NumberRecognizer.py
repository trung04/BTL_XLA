import numpy as np
import cv2
from tensorflow.keras.models import load_model

class NumberRecognizer:
    def __init__(self, model_path):
        """
        Khởi tạo đối tượng với model đã huấn luyện.
        """
        self.model = load_model(model_path)

    # ------------------------------
    # 1️⃣ Chuyển BGR → GRAYSCALE thủ công
    # ------------------------------
    def bgr_to_gray_manual(self, image):
        height, width, channels = image.shape
        gray = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                B, G, R = image[y, x]
                gray[y, x] = int(0.114 * B + 0.587 * G + 0.299 * R)
        return gray

    # ------------------------------
    # 2️⃣ Resize thủ công (nội suy trung bình)
    # ------------------------------
    def resize_manual(self, image, new_width=28, new_height=28):
        old_height, old_width = image.shape[:2]
        scale_x = old_width / new_width
        scale_y = old_height / new_height

        if len(image.shape) == 3:
            channels = image.shape[2]
            resized = np.zeros((new_height, new_width, channels), dtype=np.uint8)
        else:
            resized = np.zeros((new_height, new_width), dtype=np.uint8)

        for y in range(new_height):
            for x in range(new_width):
                x_start = int(x * scale_x)
                y_start = int(y * scale_y)
                x_end = min(int((x + 1) * scale_x), old_width)
                y_end = min(int((y + 1) * scale_y), old_height)

                region = image[y_start:y_end, x_start:x_end]
                resized[y, x] = np.mean(region, axis=(0, 1))

        return resized

    # ------------------------------
    # 3️⃣ Xử lý ảnh đầu vào → model input
    # ------------------------------
    def preprocess(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")

        # Resize và chuyển xám thủ công
        image = self.resize_manual(image, 28, 28)
        image_gray = self.bgr_to_gray_manual(image)

        # Chuyển thành dạng model chấp nhận (28, 28, 1)
        image_prediction = np.reshape(image_gray, (28, 28, 1))

        # Chuẩn hóa về [0,1] và đảo ngược trắng/đen
        image_prediction = (255 - image_prediction.astype('float')) / 255.0
        return image_prediction

    # ------------------------------
    # 4️⃣ Dự đoán
    # ------------------------------
    def predict(self, image_path):
        image_preprocessed = self.preprocess(image_path)
        preds = self.model.predict(np.array([image_preprocessed]), verbose=0)
        label = np.argmax(preds, axis=-1)[0]
        confidence = float(np.max(preds))
        return label, confidence
