import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import random

# --- 1. Tải và xử lý dữ liệu MNIST ---
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Chuẩn hóa dữ liệu về [0, 1]
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Thêm chiều kênh màu (1 kênh - ảnh xám)
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# --- 2. Hàm tạo model ---
def create_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')  # ✅ dùng 'softmax' thay vì tf.nn.softmax
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# --- 4. Tăng cường dữ liệu (Data Augmentation) ---
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.2
)

# Tạo luồng dữ liệu
train_generator = datagen.flow(train_images, train_labels, batch_size=64)
test_generator = datagen.flow(test_images, test_labels, batch_size=64)

# --- 5. Huấn luyện mô hình cải tiến ---
improved_model = create_model()
improved_model.fit(
    train_generator,
    epochs=15,
    validation_data=test_generator
)

# --- 6. Lưu mô hình (định dạng mới nhất) ---
improved_model.save("model/mnist_model.keras")
print("✅ Model saved successfully as mnist_model.keras")

# --- 7. Nạp lại mô hình để kiểm tra ---
reloaded_model = keras.models.load_model("model/mnist_model.keras")
loss, acc = reloaded_model.evaluate(test_images, test_labels, verbose=0)
print(f"✅ Reloaded model accuracy: {acc:.4f}")
