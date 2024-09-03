import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from mpl_toolkits.basemap import Basemap  
from shapely.geometry import MultiLineString, MultiPolygon, Polygon, box, shape
from matplotlib.patches import Polygon as PltPolygon

from sentinelhub import CRS

import cv2
import numpy as np
import os
import random
from PIL import Image


def plot_confusion_matrix(y_true, y_pred, crop_mapping):
    '''
    Plots a confusion matrix with count and percentage for each class.
    
    Inputs:
    - y_true: True labels
    - y_pred: Predicted labels
    - crop_mapping: Dictionary mapping crop IDs to crop names

    Output:
    - Displays the confusion matrix plot
    '''
    conf_matrix = confusion_matrix(y_true.tolist(), y_pred.tolist())
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    labels = np.asarray([['{}\n{:.2f}%'.format(conf_matrix[i, j], conf_matrix_percent[i, j]) for j in range(conf_matrix.shape[1])] for i in range(conf_matrix.shape[0])])

    plt.figure(figsize=(10, 8))
    cmap = sns.light_palette("green", as_cmap=True)
    sns.heatmap(conf_matrix_percent, annot=labels, fmt='', cmap=cmap, cbar=False,
                xticklabels=[crop_mapping[i] for i in sorted(crop_mapping.keys())],
                yticklabels=[crop_mapping[i] for i in sorted(crop_mapping.keys())])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix with Count and Percentage')
    plt.show()

def plot_crop_predictions(y_true, y_pred, crop_mapping):
    '''Plots the actual, predicted, correct, and false predictions for each crop type.
    
    Inputs:
    - y_true: True labels
    - y_pred: Predicted labels
    - crop_mapping: Dictionary mapping crop IDs to crop names

    Output:
    - Displays the crop predictions plot
    '''
    fig, axs = plt.subplots(1, len(crop_mapping), figsize=(20, 6))

    for i, (crop_id, crop_name) in enumerate(crop_mapping.items()):
        y_test_count = np.sum(y_true == crop_id)
        test_predictions_count = np.sum(y_pred == crop_id)
        correct_predictions_count = np.sum((y_true == crop_id) & (y_pred == crop_id))
        false_predictions_count = test_predictions_count - correct_predictions_count

        axs[i].bar('Actual', y_test_count, color='red', alpha=0.5)
        axs[i].bar('Predicted', test_predictions_count, color='green', alpha=0.5)
        axs[i].bar('Correct', correct_predictions_count, color='blue', alpha=0.5)
        axs[i].bar('False', false_predictions_count, color='orange', alpha=0.5)

        axs[i].set_title(crop_name)

    for ax in axs:
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.show()



def plot_crop_type_count(new_df):
    '''
    Plots the count of each crop type in the given DataFrame.
    
    Input:
    - new_df: DataFrame containing 'Field_Id' and 'Crop_Type' columns

    Output:
    - Displays the crop type count plot
    '''
    field_ids = new_df['Field_Id'].unique()
    count_values = new_df.groupby('Crop_Type')['Field_Id'].unique().apply(len).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    count_values.plot(kind='bar')
    plt.title('Décompte des valeurs de la colonne crop_type')
    plt.xlabel('Types de cultures')
    plt.ylabel('Décompte')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.show()






def show_splitter(splitter, alpha=0.2, area_buffer=0.2, show_legend=False):
    '''Visualizes the splitting of an area into bounding boxes.
    Inputs:
    - splitter: BBoxSplitter object
    - alpha: Transparency of the bounding box overlays (default: 0.2)
    - area_buffer: Buffer size around the area bbox (default: 0.2)
    - show_legend: Boolean indicating whether to show the legend (default: False)

    Output:
    - Displays the splitter visualization plot
        '''
    area_bbox = splitter.get_area_bbox()
    minx, miny, maxx, maxy = area_bbox
    lng, lat = area_bbox.middle
    w, h = maxx - minx, maxy - miny
    minx = minx - area_buffer * w
    miny = miny - area_buffer * h
    maxx = maxx + area_buffer * w
    maxy = maxy + area_buffer * h

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    base_map = Basemap(
        projection="mill",
        lat_0=lat,
        lon_0=lng,
        llcrnrlon=minx,
        llcrnrlat=miny,
        urcrnrlon=maxx,
        urcrnrlat=maxy,
        resolution="l",
        epsg=4326,
    )
    base_map.drawcoastlines(color=(0, 0, 0, 0))

    area_shape = splitter.get_area_shape()

    if isinstance(area_shape, Polygon):
        polygon_iter = [area_shape]
    elif isinstance(area_shape, MultiPolygon):
        polygon_iter = area_shape.geoms
    else:
        raise ValueError(f"Geometry of type {type(area_shape)} is not supported")

    for polygon in polygon_iter:
        ax.add_patch(
            PltPolygon(np.array(polygon.exterior.coords), closed=True, facecolor=(0, 0, 0, 0), edgecolor="red")
        )

    bbox_list = splitter.get_bbox_list()
    info_list = splitter.get_info_list()

    cm = plt.get_cmap("jet", len(bbox_list))
    legend_shapes = []
    for i, bbox in enumerate(bbox_list):
        wgs84_bbox = bbox.transform(CRS.WGS84).get_polygon()

        tile_color = tuple(list(cm(i))[:3] + [alpha])
        ax.add_patch(PltPolygon(np.array(wgs84_bbox), closed=True, facecolor=tile_color, edgecolor="green"))

        if show_legend:
            legend_shapes.append(plt.Rectangle((0, 0), 1, 1, fc=cm(i)))

    if show_legend:
        legend_names = []
        for info in info_list:
            legend_name = "{},{}".format(info["index_x"], info["index_y"])

            for prop in ["grid_index", "tile"]:
                if prop in info:
                    legend_name = "{},{}".format(info[prop], legend_name)

            legend_names.append(legend_name)

        plt.legend(legend_shapes, legend_names)
    plt.tight_layout()
    plt.show()
    
    
    
    
def process_display_image(model_image, mask_dir, output_dir):
    """
    Process and display a single image with masks applied.

    model_image : PIL image object
    mask_dir : directory containing mask files (output of the model)
    output_dir : directory to save the processed image
    """
    os.makedirs(output_dir, exist_ok=True)
    image = np.array(model_image)
    if image.ndim == 2:  
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:  
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    
    print(f"Image shape: {image.shape}")

    
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.txt')]
    
    if not mask_files:
        print(f"No mask files found in {mask_dir}")
        return

    
    mask_file = mask_files[0]
    mask_path = os.path.join(mask_dir, mask_file)

    with open(mask_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8 or len(parts) % 2 != 1:
            print(f"Format de ligne incorrect dans le fichier {mask_path}: {line}")
            continue

        coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
        coords[:, 0] *= image.shape[1]
        coords[:, 1] *= image.shape[0]
        coords = coords.astype(np.int32)

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.polylines(image, [coords], isClosed=True, color=color, thickness=2)

    output_path = os.path.join(output_dir, 'image_with_masks.jpg')
    cv2.imwrite(output_path, image)
    print(f"Image sauvegardée avec les masques: {output_path}")

    original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(model_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(original_image_rgb)
    axes[1].set_title("Image with Masks")
    axes[1].axis('off')

    plt.show()

    print("Traitement terminé.")



def adjust_color_intensity(color, intensity=0.4):
    return (color * intensity + 255 * (1 - intensity)).astype(np.uint8)

def generate_colored_masks(results, model_image, save_dir, mask_name):
    """
    this function returns masks with different colors to each object for better visulisation 
    params:
        results : results of the model on the predict methode of the model
        model_image : image returned from the model (PIL image)
        save_dir : the directory to save the colored mask 
        mask_name : the name of the colored mask 
    """

    os.makedirs(save_dir, exist_ok=True)
    
    image_array = np.array(model_image)
    image_array = image_array[:, :, ::-1]  
    original_image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_height, image_width = original_image_rgb.shape[:2]

    num_masks = len(results[0].masks.data)  
    colors = np.random.randint(0, 255, size=(num_masks, 3), dtype=np.uint8)
    intensity = 0.5
    adjusted_colors = [adjust_color_intensity(color, intensity) for color in colors]


    for j, mask in enumerate(results[0].masks.data):
        mask_resized = cv2.resize(mask.numpy().astype(np.uint8), (image_width, image_height))
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(original_image_rgb, contours, -1, color=adjusted_colors[j].tolist(), thickness=cv2.FILLED)
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(original_image_rgb)
    axes[1].set_title("Image with Colored Masks")
    axes[1].axis('off')
    plt.show()
    save_path = os.path.join(save_dir, mask_name)
    cv2.imwrite(save_path, cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR))

    print(f"Saved colored mask as {mask_name}")

import geopandas as gpd
import simplekml

def convert_to_kml(geodataframe, kml_file_path):
    """
    Convert a GeoDataFrame to a KML file.
    :param geodataframe: GeoDataFrame containing the polygons
    :param kml_file_path: Path to the output KML file
    """

    geodataframe = geodataframe[['geometry']]
    kml = simplekml.Kml()
    for index, row in geodataframe.iterrows():
        polygon = kml.newpolygon(
            name=str(index),
            outerboundaryis=list(row['geometry'].exterior.coords)
        )
        polygon.style.polystyle.color = simplekml.Color.changealphaint(0, simplekml.Color.white) 
        polygon.style.linestyle.color = simplekml.Color.green 
        polygon.style.linestyle.width = 2 
    
    kml.save(kml_file_path)
    
    
def display_image(image_path):
    img = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image (Largeur: {img.width}px, Hauteur: {img.height}px)")
    plt.show()    