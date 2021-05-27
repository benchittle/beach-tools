"""

ShorelineDelineation.py

This program takes a series of satellite/aerial images and extracts shoreline polygons from them.
The script is ideal for extracting shoreline data from historical imagery for time-series analyses

To ensure this script runs optimally, take the following steps:
    
    1. The input imagery should distinguish between land and water as much as possible
        - The optimal band combination for separating land and water recommended by ESRI is (R,G,B) = (B5,B6,B4) or (B6,B5,B2) [landsat 8] 
        - True colour imagery may be used, but classification results may vary depending on feature contrast level
        - If forced to use true colour imagery, focus on applying a strategic mask (maximize contrast between water and land beforehand)
    
    2. The mask should cover only the shoreline and water, making it easier for the classifier to create and distinguish only 2 classes
        - Within the mask, you should only be able to see two colours: land and water
        - Try to avoid including vegetation, buildings, etc. from within the mask when possible

Developed by: Valerie Morin (May 2021) [UWindsor Coastal Research Group]

"""

from os import listdir
from os.path import isfile, join
import arcpy
from arcpy import env
from arcpy.sa import *


def main():
    # Update directory and feature paths here
    input_imgs_path = r"C:/Users/ValerieMorin/Desktop/example_dir/input_imgs/"
    output_feature_path = r"C:/Users/ValerieMorin/Desktop/example_dir/output_features/"
    mask_path = r"C:/Users/ValerieMorin/Desktop/example_dir/mask/mask.shp"

    env.workspace = input_imgs_path
    arcpy.CheckOutExtension("Spatial")

    raster_list = arcpy.ListRasters()

    for i in range(len(raster_list)):
        print "\nProcessing image '" + raster_list[i] + "'. " + str(i + 1) + " of " + str(len(raster_list))

        # Clip mask --> classify clip --> convert classified raster to polygons :D
        clip_mask_from_img(raster_list[i], mask_path)
        classify_raster(raster_list[i])
        convert_raster_to_polygon(raster_list[i], output_feature_path)


# Clip the shoreline mask from the current raster object
def clip_mask_from_img(img, mask):
    print "Clipping mask from image..."
    mask_descr = arcpy.Describe(mask)
    mask_extent = "%s %s %s %s" % (mask_descr.extent.XMin, mask_descr.extent.YMin, mask_descr.extent.XMax, mask_descr.extent.YMax)
    arcpy.Clip_management(img, mask_extent, "clip_" + img, mask, arcpy.Raster(img).noDataValue, "ClippingGeometry")


# Classify the clipped raster into two classes (land and water) using Iso-Data unsupervised classifier
def classify_raster(img):
    print "Classifying clipped raster..."
    classified_raster = IsoClusterUnsupervisedClassification("clip_" + img, 2)
    classified_raster.save("class_" + img)


# Convert the classified raster into polygons, from here user can extract shoreline polygon(s)
def convert_raster_to_polygon(img, output_feature_path):
    print "Converting raster to polygon..."
    arcpy.RasterToPolygon_conversion("class_" + img, output_feature_path + img.split('.', 1)[0] + ".shp", "SIMPLIFY")


main()
