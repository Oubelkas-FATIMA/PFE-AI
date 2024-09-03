from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
from utils.annotations_utils import draw_detectronv2_annotations, draw_maskrcnn_annotations, convert_prediction_detectronv2_to_json , save_annotations_yolov_tojson , convert_prediction_mask_rcnn_to_json
from config.config import detectron_PATH , model_rcnn_path
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import random
from ultralytics import YOLO
import numpy  as  np
import matplotlib.pyplot as plt
import os


def run_inference_detectronv2(image_path, output_folder):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 27
    cfg.MODEL.WEIGHTS = detectron_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)
    im = cv2.imread(image_path)
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    convert_prediction_detectronv2_to_json(instances, image_path, output_folder)
    draw_detectronv2_annotations(instances, image_path, output_folder, show_image=True)

def predict_image(model, image_path, transform):
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        outputs = model(image_tensor)
        prediction = {
            'image_name': os.path.basename(image_path),
            'masks': outputs[0]['masks'],
            'scores': outputs[0]['scores']
        }
        return prediction

def run_inference_mask_rcnn(image_path, output_folder, model_rcnn_path):
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn(weights=weights)
    num_classes = 2  # Background + parcel
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load(model_rcnn_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    prediction = predict_image(model, image_path, transform)
    output_path = os.path.join(output_folder, "annotations.json")
    convert_prediction_mask_rcnn_to_json(prediction, output_path)
    #draw_maskrcnn_annotations(prediction, image_path, output_folder, show_image=True)
    
    
def run_inference_yolo(image_path , model_path):
    model = YOLO(model_path)
    results = model.predict(image_path, save = True, save_txt = True)
    save_annotations_yolov_tojson(results)
    img_array = np.array(image_path)
    for obj in results:
        if 'segmentation' in obj and obj['segmentation']:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            mask = np.array(obj['segmentation'], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_array, [mask], isClosed=True, color=color, thickness=2)  
'''        # Display the image
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Image with Segmentation Masks')
        plt.show()  '''
                