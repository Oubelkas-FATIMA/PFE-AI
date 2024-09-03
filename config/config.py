# config.py

import os
from dotenv import load_dotenv
from sentinelhub import SHConfig

# Load environment variables from .env file
load_dotenv()



MODEL_PARAMS = {'n_estimators': 100, 'random_state': 42}
MODEL_PATH = ""
CROP_MAPPING = {
    1: 'Cotton',
    2: 'Dates',
    3: 'Grass',
    4: 'Lucern',
    5: 'Maize',
    6: 'Pecan',
    7: 'Vacant',
    8: 'Vineyard',
    9: 'Vineyard & Pecan ("Intercrop")'
}


# Assign environment variables to Sentinel Hub configuration settings
def get_sentinel_hub_config():
    config = SHConfig()
    config.sh_client_id = os.getenv("SH_CLIENT_ID")
    config.sh_client_secret =  os.getenv("SH_CLIENT_SECRET")
    return config

EVALSCRIPT_ALL_BANDS = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],
            units: "DN"
        }],
        output: {
            bands: 12,
            sampleType: "INT16"
        }
    };
}

function evaluatePixel(sample) {
    return [sample.B01,
            sample.B02,
            sample.B03,
            sample.B04,
            sample.B05,
            sample.B06,
            sample.B07,
            sample.B08,
            sample.B8A,
            sample.B09,
            sample.B11,
            sample.B12];
}
"""

OUTPUT_FOLDER = '/Users/Hiba/satellite-imagery-pfe/data/data_sentinel'
YEAR = 2023
PROVIDER = 'sentinel'
N_CHUNKS = 2

################

EPSG_UTM =32629
TILE_SIZE = 20
EPSG_UTM_1 = "EPSG:32629"
EPSG_4326 ="EPSG:4326"


# mapbox
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
TILESET_ID = "mapbox.satellite"   
IMAGE_FORMAT = "png" 

DEFAULT_SIZE = "1224x1224" 
BASE_URL_MAPBOX = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static" 
MAPBOX_TOKEN = 'pk.eyJ1IjoiaGliYWxiIiwiYSI6ImNsdm9nMXVtYTBqODUyam8xNWVucWNtZXUifQ.j2dv4_4_rcZFyINA4wzJ8g'
DEFAULT_WIDTH_KM = 1.5 
DEFAULT_HEIGHT_KM = 1.5 


DEFAULT_OUTPUT_DIR = "ZOOM17"  
DEFAULT_ZOOM = 16 
DEFAULT_SOURCE = "Satellite"  
DEFAULT_OVERWRITE = True 
DEFAULT_SQUARE_SIZE_KM = 2 

#  Google Maps
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")  
DEFAULT_SCALE = 2  
DEFAULT_MAP_TYPE = "satellite"  
DEFAULT_SIZE = "1224x1224"  
BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"


# geometry
DEFAULT_WIDTH_KM = 1.5
DEFAULT_HEIGHT_KM = 1.5
DEFAULT_SQUARE_SIZE_KM = 2


import geopandas as gpd
gdf = gpd.read_file("C:/Users/XPRISTO/OneDrive/Bureau/livrables de stages/tourbav4/tourba.shp")


#parametres to convert image annotation to shapefile
image_width = 1224
image_height = 1224

#### modeles segmentation  
detectron_PATH = '/Users/Hiba/satellite-imagery-pfe/data/model_final.pth'
model_rcnn_path = '/Users/Hiba/satellite-imagery-pfe/data/model_checkpoint-10.pth'
model_yolo =  '/Users/Hiba/satellite-imagery-pfe/data/bestyolov9.pt'