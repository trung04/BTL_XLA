import cv2
import numpy as np
import os
import random

# Thư mục lưu dữ liệu
os.makedirs('data/circle', exist_ok=True)
os.makedirs('data/rectangle', exist_ok=True)

colors = [(0,0,0), (255,255,255), (0,0,255), (0,255,0), (255,0,0),
          (255,255,0), (0,255,255), (255,0,255)]

def random_bg(size=(64,64,3)):
    bg = np.ones(size, np.uint8) * random.randint(100,255)
    # thêm nhiễu nhẹ
    noise = np.random.randint(0,20, size, np.uint8)
    return cv2.add(bg, noise)

# --- Sinh hình tròn ---
for i in range(1000):
    img = random_bg()
    center = (random.randint(15,48), random.randint(15,48))
    radius = random.randint(8,18)
    color = random.choice(colors)
    cv2.circle(img, center, radius, color, -1)
    cv2.imwrite(f"data/circle/circle_{i}.png", img)

# --- Sinh hình vuông/chữ nhật ---
for i in range(1000):
    img = random_bg()
    x1, y1 = random.randint(10,25), random.randint(10,25)
    x2, y2 = random.randint(35,55), random.randint(35,55)
    color = random.choice(colors)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, -1)
    cv2.imwrite(f"data/rectangle/rect_{i}.png", img)

print("✅ Đã sinh dữ liệu đa dạng cho model.")