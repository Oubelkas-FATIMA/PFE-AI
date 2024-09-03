from shapely.geometry import shape, MultiPolygon
from sentinelhub import (
    CRS,
    BBoxSplitter,
)


import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
from shapely.ops import transform



class BBoxspliter:
    @staticmethod
    def bboxsplitter_epsg(shape_area, n, m, EPSG_UTM=None):
        '''
        Splits a shape area into a grid of bounding boxes.
        
        Input:
            shape_area (GeoDataFrame): The area to be split
            n (int): Number of splits in the x-direction
            m (int): Number of splits in the y-direction
            EPSG_UTM (int, optional): EPSG code for UTM projection
        
        Output:
            BBoxSplitter: A BBoxSplitter object containing the split bounding boxes
        '''
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
            bbox_splitter = BBoxSplitter([union_polygon], CRS.WGS84, (n, m))
            return bbox_splitter
        except Exception as e:
            print("An error occurred:", str(e))
            return None, None


class GeoDataFrameCreator:
    def __init__(self, src_crs, dst_crs):
        '''
        Initializes the GeoDataFrameCreator with source and destination coordinate reference systems.
        
            src_crs (str or int): EPSG code of the source coordinate system
            dst_crs (str or int): EPSG code of the destination coordinate system

        '''
        self.src_crs = src_crs
        self.dst_crs = dst_crs

    def bbox_to_polygon(self, bbox):
        '''
        Converts a bounding box to a polygon and transforms it to the destination CRS.
        
        Input:
            bbox (tuple): Bounding box coordinates (minx, miny, maxx, maxy)
        
        Output:
            Polygon: Transformed polygon object
        '''

        minx, miny, maxx, maxy = bbox
        bbox_polygon = box(minx, miny, maxx, maxy)

        def transform_func(x, y, z=None):
            return Transformer.from_crs(self.src_crs, self.dst_crs, always_xy=True).transform(x, y, z)
        
        bbox_polygon_transformed = transform(transform_func, bbox_polygon)
        
        return bbox_polygon_transformed
    

    def create_geo_dataframe(self, bbox_list, filename=None):
        '''
        Creates a GeoDataFrame from a list of bounding boxes.
        
        Input:
            bbox_list (list): List of bounding box coordinates
            filename (str, optional): Path to save the GeoDataFrame as a file
        
        Output:
            GeoDataFrame: Created GeoDataFrame with transformed polygons
        '''
        transformed_polygons = [
            self.bbox_to_polygon(bbox)
            for bbox in bbox_list
        ]
        
        gdf = gpd.GeoDataFrame(geometry=transformed_polygons)
        gdf.crs = self.dst_crs
        
        if filename:
            gdf.to_file(filename)
        
        return gdf

