import os
import cv2
import numpy as np
from pycocotools import mask as mask_util
import matplotlib.pyplot as plt
import random
import json

import json
import numpy as np
from pycocotools import mask as mask_util



def convert_prediction_detectronv2_to_json(instances, image_path, output_file):
    '''
    Converts the predicted instances to COCO JSON format and saves it to a file.

    Inputs:
    - instances: Predicted instances from the model
    - image_path: Path to the input image
    - output_file: Path to save the JSON output
    '''
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "classe1"},
            {"id": 2, "name": "classe2"},
        ]
    }

    image_info = {
        "id": 1,
        "file_name": image_path,
        "width": instances.image_size[1],
        "height": instances.image_size[0]
    }
    coco_output["images"].append(image_info)

    for i in range(len(instances)):
        box = instances.pred_boxes.tensor[i].cpu().numpy()
        score = instances.scores[i].item()
        category_id = instances.pred_classes[i].item() + 1
        mask = instances.pred_masks[i].cpu().numpy()

        mask = mask.astype(np.uint8)
        mask = mask_util.decode(mask_util.encode(np.asarray(mask, order="F")))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) > 4:
                segmentation.append(contour)

        annotation_info = {
            "id": i + 1,
            "image_id": 1,
            "category_id": category_id,
            "bbox": box.tolist(),
            "score": score,
            "segmentation": segmentation,
            "area": float(mask_util.area(mask_util.encode(np.asarray(mask, order="F")))),
            "iscrowd": 0
        }
        coco_output["annotations"].append(annotation_info)

    with open(output_file, "w") as f:
        json.dump(coco_output, f)
    print(f"Annotation prédite sauvegardée dans '{output_file}'")

def convert_prediction_mask_rcnn_to_json(prediction, output_path):
    image_id = prediction['image_name']
    masks_tensor = prediction['masks']
    scores_tensor = prediction['scores']
    masks_np = masks_tensor.numpy()
    scores_np = scores_tensor.numpy()

    if len(masks_np.shape) == 3:
        num_masks, height, width = masks_np.shape
    else:
        num_masks, _, height, width = masks_np.shape

    annotations = []
    for i in range(num_masks):
        if len(masks_np.shape) == 3:
            mask = masks_np[i, :, :]
        else:
            mask = masks_np[i, 0, :, :]
        score = scores_np[i]
        binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, w, h]
            area = cv2.contourArea(contour)
            segmentation = contour.flatten().tolist()
            segmentation = [segmentation]
            if score >= 0.20 or area >= 30:
                annotation_info = {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "score": float(score),
                    "segmentation": segmentation,
                    "area": float(area),
                    "iscrowd": 0
                }
                annotations.append(annotation_info)

    coco_output = {
        "images": [
            {
                "id": image_id,
                "file_name": prediction['image_name'],
                "width": width,
                "height": height
            }
        ],
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "classe1"},
            {"id": 2, "name": "classe2"}
        ]
    }

    with open(output_path, "w") as f:
        json.dump(coco_output, f)
    print(f"Predicted annotation saved in '{output_path}'")    

def save_annotations_to_json(instances, image_path, output_folder):
    '''
    Removes overlapping annotations based on the specified overlap threshold and minimum area.

    Inputs:
    - annotations: List of annotations
    - overlap_threshold: Threshold for determining significant overlap (default: 0.85)
    - min_area: Minimum area for an annotation to be considered (default: 1000.0)

    Output:
    - non_overlapping_annotations: List of annotations with overlapping annotations removed
    '''
    json_path = os.path.join(output_folder, "annotations.json")
    convert_prediction_detectronv2_to_json(instances, image_path, json_path)





def remove_overlapping_annotations(annotations, overlap_threshold=0.85, min_area=1000.0):
    '''
    calculate_overlap_ratio(box1, box2)

    Calculates the overlap ratio between two bounding boxes.

    Inputs:
    - box1: First bounding box (x, y, width, height)
    - box2: Second bounding box (x, y, width, height)

    Output:
    - overlap_ratio: The overlap ratio between the two bounding boxes
    '''
    sorted_annotations = sorted(annotations, key=lambda x: x['score'], reverse=True)
    non_overlapping_annotations = []
    for annotation in sorted_annotations:
        area = annotation['area']
        if area < min_area:
            continue
        overlaps_significantly = False
        for added_annotation in non_overlapping_annotations:
            overlap_ratio = calculate_overlap_ratio(annotation['bbox'], added_annotation['bbox'])
            if overlap_ratio > overlap_threshold:
                overlaps_significantly = True
                if annotation['score'] > added_annotation['score']:
                    non_overlapping_annotations.remove(added_annotation)
                else:
                    break
        if not overlaps_significantly:
            non_overlapping_annotations.append(annotation)
    return non_overlapping_annotations

def calculate_overlap_ratio(box1, box2):
    '''
    draw_annotations(instances, image_path, output_folder, show_image=False)

    Draws the predicted annotations on the input image and saves the annotated image to the output folder.

    Inputs:
    - instances: Predicted instances from the model
    - image_path: Path to the input image
    - output_folder: Folder to save the annotated image
    - show_image: Flag to display the annotated image (default: False)
    
    '''
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = x_overlap * y_overlap
    area1 = w1 * h1
    area2 = w2 * h2
    overlap_ratio = overlap_area / min(area1, area2)
    return overlap_ratio

def draw_detectronv2_annotations(instances, image_path, output_folder ,show_image=False):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    annotations = []
    for i in range(len(instances)):
        mask = instances.pred_masks[i].cpu().numpy()
        mask = mask.astype(np.uint8)
        mask = mask_util.decode(mask_util.encode(np.asarray(mask, order="F")))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) > 4:
                bbox = cv2.boundingRect(contour)
                area = bbox[2] * bbox[3]
                score = float(instances.scores[i])
                annotation = {
                    'bbox': bbox,
                    'area': area,
                    'score': score,
                    'contour': contour
                }
                annotations.append(annotation)

    non_overlapping_annotations = remove_overlapping_annotations(annotations)

    for annotation in non_overlapping_annotations:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.polylines(image, [annotation['contour'].reshape((-1, 1, 2))], isClosed=True, color=color, thickness=2)

    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(os.path.basename(image_path))
    plt.savefig(output_image_path, bbox_inches='tight')

    if show_image:
        plt.show()
    else:
        plt.close()
        
        
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from pycocotools import mask as mask_util

def draw_maskrcnn_annotations(prediction, image_path, output_folder, show_image=False):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    annotations = []
    masks_np = prediction['masks'].numpy()
    scores_np = prediction['scores'].numpy()

    print("Mask shape:", prediction['masks'].shape)

    num_masks = masks_np.shape[0]
    for i in range(num_masks):
        mask = masks_np[i]
        mask = mask.squeeze()  # Supprime les dimensions inutiles
        mask = mask.astype(np.uint8)  # Convertit le masque en type uint8
        mask = mask_util.decode(mask_util.encode(np.asarray(mask, order="F")))  # Décode le masque
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print("Number of contours:", len(contours))

        for contour in contours:
            if len(contour) > 4:
                bbox = cv2.boundingRect(contour)
                area = bbox[2] * bbox[3]
                score = float(scores_np[i])
                annotation = {
                    'bbox': bbox,
                    'area': area,
                    'score': score,
                    'contour': contour
                }
                annotations.append(annotation)

    non_overlapping_annotations = remove_overlapping_annotations(annotations)

    for annotation in non_overlapping_annotations:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.polylines(image, [annotation['contour'].reshape((-1, 1, 2))], isClosed=True, color=color, thickness=2)

    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if show_image:
        cv2.imshow("Annotated Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()       
        
        
def save_annotations_yolov_tojson(yolo_results_list, image_id: int = 1):
    annotations = []
    
    if isinstance(yolo_results_list, list):
        yolo_results = yolo_results_list[0]
    else:
        yolo_results = yolo_results_list
    
    boxes = yolo_results.boxes
    masks = yolo_results.masks
    
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
        width = xmax - xmin
        height = ymax - ymin
        bbox = [float(xmin), float(ymin), float(width), float(height)]

        score = float(box.conf[0].cpu().numpy())       
        category_id = int(box.cls[0].cpu().numpy())
        contours = mask.xy[0].tolist()
        
        segmentation = [float(coord) for point in contours for coord in point]
        
       
        area = float(width * height)
        
        annotation = {
            "id": i + 1,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "score": score,
            "segmentation": [segmentation],
            "area": area
        }
        
        annotations.append(annotation)
    

    result_json = {
        "images": [{
            "id": image_id,
            "file_name": os.path.basename(yolo_results.path),
            "width": yolo_results.orig_img.shape[1],
            "height": yolo_results.orig_img.shape[0]
        }],
        "annotations": annotations
    }
    
    return result_json        


