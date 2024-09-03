import streamlit as st
import requests
import base64
from PIL import Image
import io
import json
import cv2
import numpy as np
import random


"""
This file exposes at the port choosen (in this case 5000) a web interface that dispalys the images predicted with results  using streamlit  
"""

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def send_request(image):
    base64_image = encode_image(image)
    json_payload = {"image": f"data:image/png;base64,{base64_image}"}
    
    response = requests.post("http://127.0.0.1:5000/predict", json=json_payload)
    return response.json()

def draw_detections(image, results):
    img_array = np.array(image)
    for obj in results:
        # Draw bounding box
        # if 'bbox' in obj:
        #     bbox = obj['bbox']
        #     x1, y1, x2, y2 = [int(coord) for coord in bbox]
        #     color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #     cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
        
        # Draw segmentation mask
        if 'segmentation' in obj and obj['segmentation']:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            mask = np.array(obj['segmentation'], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_array, [mask], isClosed=True, color=color, thickness=2)
    
    return Image.fromarray(img_array)

def main():
    st.title("YOLOv9 Object Detection and Segmentation")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Segmentation Visualization'):
            with st.spinner('Detecting objects...'):
                results = send_request(image)
            
            st.json(results)

            annotated_image = draw_detections(image, results)
            st.image(annotated_image, caption='Segmentation Visualization', use_column_width=True)

            # Save the annotated image
            buffered = io.BytesIO()
            annotated_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            href = f'<a href="data:file/png;base64,{img_str}" download="annotated_image.png">Download Annotated Image</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()