from flask import Flask, render_template, request, jsonify
from roboflow import Roboflow
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import os
import io
import json

app = Flask(__name__, static_url_path='/static')

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'result'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def predict_from_roboflow(image, model):
    api_key = "32RIu2RceoKcVPqdLQ75"

    roboflow = Roboflow(api_key=api_key)
    project = roboflow.workspace().project("wood-defect-detection-9w3x4")
    model = project.version(3).model

    predictions = model.predict(image)
    # infer on a local image
    print(model.predict(image, confidence=40, overlap=30).json())

    return predictions

def draw_annotations(image, predictions):
    if predictions and isinstance(predictions, dict) and "predictions" in predictions:
        for prediction in predictions["predictions"]:
            draw_box(image, prediction)

def draw_box(image, prediction):
    if isinstance(prediction, dict) and 'bbox' in prediction:
        bbox = prediction['bbox']
        x, y, width, height = map(int, bbox)

        color = (0, 255, 0)  # Warna kotak (hijau)
        thickness = 2  # Ketebalan garis kotak

        # Gambar kotak pada gambar asli
        cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)

        # Tambahkan label dan confidence pada kotak
        label = prediction.get('class', 'Unknown')
        confidence = prediction.get('confidence', 0.0)
        label_text = f"{label}: {confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Gambar label di atas kotak
        cv2.rectangle(image, (x, y - label_size[1] - 5), (x + label_size[0], y - 5), color, cv2.FILLED)
        cv2.putText(image, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        print("Atribut bbox tidak ditemukan dalam objek prediksi atau format prediksi tidak valid.")


model_id = "wood-defect-detection-9w3x4/3"  # Ganti dengan MODEL_ID yang sesuai
api_key = "32RIu2RceoKcVPqdLQ75"


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST', 'GET'])
def upload_image():
    # Ambil gambar yang diunggah
    if request.method == 'POST':
        uploaded_file = request.files['image']

        # Periksa apakah file yang diunggah adalah gambar
        if uploaded_file and uploaded_file.filename.endswith(('.png', '.jpg', '.jpeg')):
            filename = secure_filename(uploaded_file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(upload_path)

            # Baca gambar
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # Prediksi Gambar dari model Roboflow
            predictions = predict_from_roboflow(image_np, model_id)

            # Simpan hasil prediksi
            result_image_path = 'static/img/result/prediction.jpg'
            cv2.imwrite(result_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            return render_template('index.html', image_path=result_image_path)
        else:
            return jsonify({
                "error": "Invalid file format. Please upload a valid image."
            })
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
