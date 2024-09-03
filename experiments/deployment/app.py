from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64


"""
This file exposes endpoint http of model deployed to docker 
"""


app = Flask(__name__)

model = YOLO('bestyolov9.pt') # Path to model to use for prediction



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({"error": "No image data"}), 400
    
    image_data = request.json['image']
    image_data = base64.b64decode(image_data.split(',')[1])
    npimg = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    results = model(img)
    
    output = []
    for r in results:
        for i, (box, mask) in enumerate(zip(r.boxes, r.masks)):
            b = box.xyxy[0].tolist()
            c = box.cls
            seg = mask.xy[0].tolist() if mask is not None else None
            output.append({
                "id": i,
                "bbox": b,
                "class": model.names[int(c)],
                "confidence": float(box.conf),
                "segmentation": seg
            })
    
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
