import geopandas as gpd
from shapely.wkt import loads
import json
import math
import os
from shapely import wkt
from shapely.geometry import Polygon
from geopy.distance import distance
from config.config import DEFAULT_ZOOM, image_width, image_height 



def calculate_image_bounds(center_lat, center_lng, DEFAULT_ZOOM, image_width, image_height):
    '''
    Calculate the geographic bounds of an image based on its center coordinates, zoom level, and dimensions.
    
    :param center_lat: Latitude of the image center
    :param center_lng: Longitude of the image center
    :param zoom: Zoom level of the image
    :param image_width: Width of the image in pixels
    :param image_height: Height of the image in pixels
    :return: Tuple containing the minimum longitude, minimum latitude, maximum longitude, and maximum latitude of the image bounds
    '''
    zoom = DEFAULT_ZOOM + 1
    scale = 2 ** zoom
    earth_radius = 6378137
    radians = math.pi / 180
    x = (center_lng + 180) / 360
    y = (1 - math.log(math.tan(center_lat * radians) + 1 / math.cos(center_lat * radians)) / math.pi) / 2
    tile_size = 256
    zoom_factor = tile_size / image_width
    x_min = x - 0.5 / scale / zoom_factor
    y_min = y - 0.5 / scale / zoom_factor
    x_max = x + 0.5 / scale / zoom_factor
    y_max = y + 0.5 / scale / zoom_factor
    
    def lat_from_y(y):
        return math.atan(math.sinh(math.pi * (1 - 2 * y))) / radians
    
    def lng_from_x(x):
        return 360 * x - 180
    
    min_lat = lat_from_y(y_max)
    max_lat = lat_from_y(y_min)
    min_lng = lng_from_x(x_min)
    max_lng = lng_from_x(x_max)
    
    return (min_lng, min_lat, max_lng, max_lat)

def convert_annotations_to_polygons(image_corners, annotations_file, image_width, image_height):
    '''
    Convert annotations from pixel coordinates to geographic coordinates and create polygons.
    
    :param image_corners: Tuple containing the minimum longitude, minimum latitude, maximum longitude, and maximum latitude of the image bounds
    :param annotations_file: Path to the JSON file containing the annotations
    :param image_width: Width of the image in pixels
    :param image_height: Height of the image in pixels
    :return: GeoDataFrame containing the polygons converted from annotations
    '''
    min_lng, min_lat, max_lng, max_lat = image_corners
    with open(annotations_file) as f:
        data = json.load(f)
    annotations = data['annotations']
    polygons = []
    lng_ratio = (max_lng - min_lng) / image_width
    lat_ratio = (max_lat - min_lat) / image_height
    for annotation in annotations:
        segmentation = annotation['segmentation'][0]
        geo_points = []
        for i in range(0, len(segmentation), 2):
            x, y = segmentation[i], segmentation[i+1]
            lng = min_lng + x * lng_ratio
            lat = max_lat - y * lat_ratio
            geo_points.append((lng, lat))
        polygon = Polygon(geo_points)
        polygons.append(polygon)
    gdf = gpd.GeoDataFrame(geometry=polygons, crs='EPSG:4326')
    return gdf