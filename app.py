from flask  import Flask, jsonify,render_template,request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import re
from ShapeRecognizer import ShapeRecognizer  # import class
from NumberRecognizer import NumberRecognizer  # import class
app = Flask(__name__)
model = load_model('model/mnist_model.keras')
model2 = load_model('model/shape_cnn_model_color.h5')

@app.route('/')
def home():
    data = "hahahah"
    return render_template('index2.html',data=data)
@app.route('/recognize', methods=['POST'])
def predict():
    if request.method == 'POST':
        data_url = request.get_json()
        imageBase64 = data_url['data']  
        imgByte = base64.b64decode(imageBase64)
        with open("output.png", "wb") as f:
            f.write(imgByte)
        # Sử dụng NumberRecognizer để dự đoán
        recognizer = NumberRecognizer(model_path='mnist_model.keras')
        prediction = recognizer.predict("output.png")
        return jsonify({'prediction':str(prediction[0]),'status':   True})
    




# Trang dự đoán hình ảnh hình học
@app.route('/hinh-anh')
def hinh_anh(): 
    return render_template('hinh-anh.html')

@app.route('/recognize2', methods=['POST'])
def predic2():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    # Đọc ảnh bằng OpenCV
    img_path = "temp.jpg"
    file.save(img_path)
    recognizer = ShapeRecognizer(model_path='shape_cnn_model_color.h5', target_size=(64, 64))
    img_path= "temp.jpg"
    label, confidence = recognizer.predict(img_path)
    return jsonify({
        "shape": label,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(debug=True)