from shapely import wkt
from shapely.geometry import Polygon, shape, MultiPolygon
from shapely.validation import explain_validity
from geopy.distance import distance
from shapely.wkt import loads as load_wkt
from math import ceil
from shapely.geometry import mapping
from config.config import DEFAULT_WIDTH_KM, DEFAULT_HEIGHT_KM, DEFAULT_SQUARE_SIZE_KM

def calculate_points(centroid, half_width_km, half_height_km):
    """
    Calculate the coordinates of the points of a bounding box or square polygon
    around the centroid.

    Args:
        centroid (Point): The centroid of the polygon.
        half_width_km (float): Half of the width of the bounding box or square polygon in kilometers.
        half_height_km (float): Half of the height of the bounding box or square polygon in kilometers.

    Returns:
        Tuple[float, float, float, float]: Tuple containing the coordinates of the west, north, east, and south points respectively.
    """
    west = distance(kilometers=half_width_km).destination((centroid.y, centroid.x), bearing=270)
    north = distance(kilometers=half_height_km).destination((centroid.y, centroid.x), bearing=0)
    east = distance(kilometers=half_width_km).destination((centroid.y, centroid.x), bearing=90)
    south = distance(kilometers=half_height_km).destination((centroid.y, centroid.x), bearing=180)
    return west.longitude, north.latitude, east.longitude, south.latitude

def create_square_polygon(input_polygon_str, square_size_km=DEFAULT_SQUARE_SIZE_KM):
    """
    Create a square polygon around the centroid of the input polygon.

    Args:
        input_polygon_str (str): Well-known text representation of the input polygon.
        square_size_km (float): Size of the square polygon in kilometers. Default is DEFAULT_SQUARE_SIZE_KM.

    Returns:
        str: Well-known text representation of the square polygon.
    """
    polygon = wkt.loads(input_polygon_str)
    centroid = polygon.centroid

    half_square_size_km = square_size_km / 2
    west_lon, north_lat, east_lon, south_lat = calculate_points(centroid, half_square_size_km, half_square_size_km)

    square_points = [
        (west_lon, north_lat),
        (east_lon, north_lat),
        (east_lon, south_lat),
        (west_lon, south_lat),
        (west_lon, north_lat),
    ]

    square_polygon = Polygon(square_points)
    return square_polygon.wkt ,  (centroid.y, centroid.x)

def create_bounding_box_from_polygon(polygon_wkt, width_km=DEFAULT_WIDTH_KM, height_km=DEFAULT_HEIGHT_KM):
    """
    Create a bounding box around the input polygon.

    Args:
        polygon_wkt (str): Well-known text representation of the input polygon.
        width_km (float): Width of the bounding box in kilometers. Default is DEFAULT_WIDTH_KM.
        height_km (float): Height of the bounding box in kilometers. Default is DEFAULT_HEIGHT_KM.

    Returns:
        list: List containing the coordinates of the west, south, east, and north points respectively.
    """
    polygon = load_wkt(polygon_wkt)

    if not isinstance(polygon, Polygon):
        raise ValueError("Input is not a valid polygon.")

    centroid = polygon.centroid
    half_width = width_km / 2
    half_height = height_km / 2

    west_lon, north_lat, east_lon, south_lat = calculate_points(centroid, half_width, half_height)
    bbox = [west_lon, south_lat, east_lon, north_lat]

    return bbox

def calculate_split_shape(shape_area, tile_width_km, tile_height_km, EPSG_UTM=None):
    """
    shape_area : the shapefile that contains polygones geometry
    tile_width_km, tile_height_km : 20 km
    EPSG_UTM : epsg should be transformed to UTM to calculate the area
    """
    try:
        if EPSG_UTM is not None:
            shape_area = shape_area.to_crs(epsg=EPSG_UTM)

        geometries = shape_area.geometry
        polygons = []
        for geometry in geometries:
            if geometry is not None:
                polygon = shape(geometry)
                if polygon.is_valid:
                    polygons.append(polygon)
        union_polygon = MultiPolygon(polygons)
        minx, miny, maxx, maxy = union_polygon.bounds
        bbox_width = (maxx - minx) / 1000
        bbox_height = (maxy - miny) / 1000
        n = ceil(bbox_width / tile_width_km)
        m = ceil(bbox_height / tile_height_km)
        return n, m
    except Exception as e:
        print("An error occurred:", str(e))
        return None, None
    
    
@staticmethod

def convert_polygon(polygon_str):
    '''
    Converts a polygon in WKT format to a dictionary with coordinates.
    Inputs:
    - polygon_str: Polygon in WKT format
    
    Output:
    - Dictionary with polygon coordinates
    '''
    try:
        geometry = wkt.loads(polygon_str)
        geojson = mapping(geometry)
        return geojson
    except Exception as e:
        print(f"Error converting polygon: {e}")
        return None
    
    