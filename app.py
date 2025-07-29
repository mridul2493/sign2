import logging
logging.basicConfig(level=logging.INFO)

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from predict_utils import predict_from_frame

app = Flask(__name__)
current_word = ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global current_word
    data = request.json.get('image')
    img_data = base64.b64decode(data.split(',')[1])
    np_img = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    letter, current_word = predict_from_frame(frame, current_word)
    return jsonify({'letter': letter, 'word': current_word})

@app.route('/reset', methods=['POST'])
def reset():
    global current_word
    current_word = ""
    return jsonify({'status': 'reset'})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
