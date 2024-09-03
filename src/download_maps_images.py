import requests
import os
import logging
import io
from PIL import Image
from shapely.wkt import loads
from config.config import  BASE_URL_MAPBOX, DEFAULT_ZOOM, DEFAULT_SIZE, TILESET_ID, IMAGE_FORMAT, MAPBOX_TOKEN , GOOGLE_MAPS_API_KEY, DEFAULT_SCALE, DEFAULT_MAP_TYPE, BASE_URL, DEFAULT_OUTPUT_DIR,  DEFAULT_SOURCE, DEFAULT_OVERWRITE
import math
from utils.filename_utils import clean_filename
import urllib.parse


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Map Box APIs

class MapboxStaticTiles:
    @staticmethod
    def get_static_map(polygon_wkt, output_path):
        '''
        Download a static map image from Mapbox for a given polygon.
        Parameters:
        polygon_wkt (str): Polygon in WKT format to determine the area of interest.
        Returns:
        Image: PIL Image object of the map.
        '''
        polygon = loads(polygon_wkt)
        minx, miny, maxx, maxy = polygon.bounds
        center_lat = (miny + maxy) / 2
        center_lng = (minx + maxx) / 2
        params = {
            "access_token": MAPBOX_TOKEN,
            "center": f"{center_lng},{center_lat}",
            "zoom": DEFAULT_ZOOM,
            "size": DEFAULT_SIZE
        }
        url = f"{BASE_URL_MAPBOX}/{center_lng},{center_lat},{params['zoom']}/{params['size']}?access_token={params['access_token']}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Tile downloaded and saved successfully to {output_path}")
            return output_path
        else:
            logging.error(f"Error while downloading: {response.status_code}. URL : {url}")
            return None


            
class MapboxRasterTiles:
    @staticmethod
    def lat_lon_to_tile_xy(latitude, longitude, zoom):
        '''
        Convert latitude and longitude to tile coordinates for a given zoom level.

        Parameters:
        latitude (float): Latitude of the point.    
        longitude (float): Longitude of the point.
        zoom (int): Zoom level.

        Returns:
        tuple: Tile x and y coordinates.
        '''
        lat_rad = math.radians(latitude)
        n = 2 ** zoom
        x = int((longitude + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    @staticmethod
    def download_raster_tile(output_dir, parcel_name, index, x, y):
        '''
        Download a raster tile image from Mapbox for given tile coordinates.

        Parameters:
        output_dir (str): Directory to save the downloaded image.
        parcel_name (str): Name of the parcel for naming the image file.
        index (int): Index to differentiate multiple images.
        x (int): Tile x coordinate.
        y (int): Tile y coordinate.
        '''
        tile_url = f"https://api.mapbox.com/v4/{TILESET_ID}/{DEFAULT_ZOOM}/{x}/{y}@2x.{IMAGE_FORMAT}?access_token={MAPBOX_TOKEN}"
        
        response = requests.get(tile_url)
        
        if response.status_code == 200:
            file_name = clean_filename(parcel_name)  
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            file_path = f"{output_dir}/{index}_{file_name}.{IMAGE_FORMAT}"
            
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            logging.info(f"Tile downloaded successfully: {file_path}")
        else:
            logging.error(f"Error while downloading: {response.status_code}")



#### Google Maps API 
class GoogleMaps:
    @staticmethod
    def get_static_map(output_dir, parcel_name, index, polygon_wkt):
        '''
        Download a static map image from Google Maps for a given polygon.

        Parameters:
        output_dir (str): Directory to save the downloaded image.
        parcel_name (str): Name of the parcel for naming the image file.
        index (int): Index to differentiate multiple images.
        polygon_wkt (str): Polygon in WKT format to determine the area of interest.
        '''
        
        polygon = loads(polygon_wkt)
    
        minx, miny, maxx, maxy = polygon.bounds
        center_lat = (miny + maxy) / 2
        center_lng = (minx + maxx) / 2

        params = {
            "center": f"{center_lat},{center_lng}",
            "zoom": DEFAULT_ZOOM,  
            "scale": DEFAULT_SCALE, 
            "maptype": DEFAULT_MAP_TYPE, 
            "size": DEFAULT_SIZE,  
            "key": GOOGLE_MAPS_API_KEY  
        }

        url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"

        response = requests.get(url)
        
        if response.status_code == 200:
            file_name = clean_filename(parcel_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_path = f"{output_dir}/{index}_{file_name}.png"
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            logging.info(f"Tile downloaded successfully: {output_path}")
        else:
            logging.error(f"Error while downloading: {response.status_code}. URL : {url}")

#### SAM Geo
class GeoTif:
    @staticmethod       
    def geotif(output=None, bbox=None, zoom=DEFAULT_ZOOM, source=DEFAULT_SOURCE, overwrite=DEFAULT_OVERWRITE):
        """
        Convert TMS to GeoTIFF.

        Parameters:
        output (str): The path to the output image.
        bbox (tuple): The bounding box coordinates.
        zoom (int, optional): The zoom level. Default is 17.
        source (str, optional): The source of the tiles. Default is "Satellite".
        overwrite (bool, optional): Whether to overwrite the existing file. Default is True.
        """
        tms_to_geotiff(output=output, bbox=bbox, zoom=zoom, source=source, overwrite=overwrite)
        