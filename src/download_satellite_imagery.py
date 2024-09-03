from sentinelhub import SentinelHubRequest, MimeType, CRS, BBox, DataCollection, SHConfig
import logging
from datetime import datetime
import os
import hashlib
import requests
import ee
from shapely.geometry import mapping, shape
import re

from shapely import wkt
from utils.geometry_utils  import  convert_polygon
from shapely import wkt

##### Sentinel Hub 
class DownloadWithSentinel:
    def __init__(self, config):
        self.config = config

    def download_sentinel_images(self, bbox_splitter, output_folder, time_interval, evalscript):
        '''
        Download Sentinel-2 images for given bounding boxes and time interval.

        Args:
            bbox_splitter (BBoxSplitter): Object containing bounding boxes.
            output_folder (str): Folder to save downloaded images.
            time_interval (tuple): Start and end dates for image acquisition.
            evalscript (str): Script for processing Sentinel-2 data.

        Returns:
            None
        '''
        for bbox_info in bbox_splitter.get_bbox_list():
            bbox = BBox(bbox_info.geometry, CRS.WGS84)
            bbox_coords = str(bbox_info.geometry).replace("(", "").replace(")", "").replace(", ", "_")
            polygon_folder = os.path.join(output_folder, f"{bbox_coords}")
            if not os.path.exists(polygon_folder):
                os.makedirs(polygon_folder)
            
            size = (1250, 1250)
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval,
                        other_args={"dataFilter": {"mosaickingOrder": "leastCC"}},
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=bbox,
                size=size,
                data_folder=polygon_folder,
                config=self.config
            )
            
            try:
                response = request.get_data(save_data=True)
                logging.info(f"Image downloaded for polygon: {bbox_coords}")
            except Exception as e:
                logging.error(f"Error downloading image for polygon {bbox_coords}: {str(e)}")




### Copernicus
class DownloadWithCopernicus:
    def __init__(self, config):
        self.config = config

    def download_sentinel_images(self, bbox_splitter, output_folder, time_interval, evalscript):
        '''
        Downloads Sentinel images from Copernicus for an area defined by a bbox_splitter.
        
        Inputs:
        - bbox_splitter: Object to split an area into multiple bboxes
        - output_folder: Output folder to save the images
        - time_interval: Time interval to filter the images
        - evalscript: Evaluation script to process the images
        
        Output:
        - Saves the downloaded images in the specified folder
        '''
        for bbox_info in bbox_splitter.get_bbox_list():
            bbox = BBox(bbox_info.geometry, CRS.WGS84)

            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A.define_from(
                            name="s2l2a", service_url="https://sh.dataspace.copernicus.eu"
                        ),
                        time_interval=time_interval,
                        other_args={"dataFilter": {"mosaickingOrder": "leastCC"}},
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=bbox,
                data_folder=output_folder,
                config=self.config
            )

            response = request.get_data(save_data=True)




# Google Earth Engine


class DownloadWithGEE:
    @staticmethod
    def download_gee_image(self, polygon_str, start_date="2022-06-01", end_date="2022-08-01", cloud_percentage=5, output_dir="output"):
        '''
        Downloads an image from Google Earth Engine for a given polygon in WKT format.
        Inputs:
        - polygon_str: Polygon in WKT format defining the area of interest
        - start_date: Start date to filter the images
        - end_date: End date to filter the images
        - cloud_percentage: Maximum cloud cover percentage allowed (default: 5)
        - output_dir: Output directory to save the downloaded image (default: "output")
        
        Output:
        - Downloads the image and saves it in the specified output directory
        '''
        bbox = self.convert_polygon(polygon_str)
        if not bbox:
            raise ValueError("Invalid polygon")

        image = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                 .filterBounds(ee.Geometry.Polygon(bbox['coordinates']))
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_percentage))
                 .sort('CLOUDY_PIXEL_PERCENTAGE') 
                 .first())
        
        if image is None:
            print("No image found for the specified criteria.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        download_url = image.getDownloadURL({
                "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
                "dimensions": [720, 720],
                "region": bbox,
                "filePerBand": False,
                "crs": "EPSG:4326",
        })
        
        response = requests.get(download_url)
        tiff_filename = os.path.join(output_dir, f"image_{image_date}.tif")
        
        with open(tiff_filename, "wb") as fd:
            fd.write(response.content)
        
        print(f"Image from {image_date} downloaded to {tiff_filename}")
        
        
def download_and_structure_images(output_folder, year, provider, n_chunks, bbox_splitter, evalscript=None, config=None):
    '''
    Download and structure satellite images for a given year and provider.

    Args:
        output_folder (str): Folder to save downloaded images.
        year (int): Year for which to download images.
        provider (str): Satellite data provider ('sentinel', 'copernicus', or 'gee').
        n_chunks (int): Number of time chunks to split the year into.
        bbox_splitter (BBoxSplitter): Object containing bounding boxes.
        evalscript (str, optional): Script for processing Sentinel-2 data.
        config (SHConfig, optional): Configuration for SentinelHub API.

    Returns:
        None
    '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    chunk_duration = (end_date - start_date) // n_chunks
    time_intervals = []
    
    for i in range(n_chunks):
        chunk_start = start_date + chunk_duration * i
        chunk_end = chunk_start + chunk_duration
        time_intervals.append((chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
    
    for time_interval in time_intervals:
        collection_folder = os.path.join(output_folder, time_interval[0])
        if not os.path.exists(collection_folder):
            os.makedirs(collection_folder)
        
        if provider == "sentinel":
            downloader = DownloadWithSentinel(config)
            downloader.download_sentinel_images(bbox_splitter, collection_folder, time_interval, evalscript)
        elif provider == "copernicus":
            copernicus = DownloadWithCopernicus(config)
            for bbox_info in bbox_splitter.get_bbox_list():
                bbox_coords = str(bbox_info.geometry).replace("(", "").replace(")", "").replace(", ", "_")
                polygon_folder = os.path.join(collection_folder, bbox_coords)
                if not os.path.exists(polygon_folder):
                    os.makedirs(polygon_folder)
                copernicus.download_sentinel_images(bbox_splitter, polygon_folder, time_interval, evalscript)
        elif provider == "gee":
            gee = DownloadWithGEE()
            for bbox_info in bbox_splitter.get_bbox_list():
                polygon_wkt = bbox_info.geometry.wkt
                bbox_coords = str(bbox_info.geometry).replace("(", "").replace(")", "").replace(", ", "_")
                polygon_folder = os.path.join(collection_folder, bbox_coords)
                if not os.path.exists(polygon_folder):
                    os.makedirs(polygon_folder)
                gee.download_gee_image(polygon_wkt, time_interval[0], time_interval[1], output_dir=polygon_folder)
        else:
            raise ValueError(f"Provider '{provider}' not supported. Supported providers: 'sentinel', 'copernicus', 'gee'")        