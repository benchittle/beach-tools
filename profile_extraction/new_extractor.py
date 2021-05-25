import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import shapely
from shapely.ops import unary_union
from shapely.geometry import LineString
from shapely.geometry import shape
from shapely.geometry import mapping
import math


##Import Raster File##
##img = rasterio.open(r'C:\DTOP_2020(Summer)\New folder\Change\bc.tif')

##Import .shp to define the shoreline position (Needs to be updated to also accept .kml)
shape_file = gpd.read_file(r'C:\Users\Ben2020\Downloads\shore\shore.shp')

##Determines shape and lists the coordinates of all vertexes from the .shp file
geom = [i for i in shape_file.geometry]

###############
# The shape_file.geometry variable is a GeoPandas GeoSeries (like a List but better).
# Since it's built on top of regular Pandas, anything you could do to a Pandas 
# Series can also be done to a GeoPandas GeoSeries
# Its entries are LineString values here (from the shapefile):
print("\nFirst (and only) entry in the shape_file.geometry GeoSeries:", shape_file.geometry.iat[0])
# It looks like the first and only entry in this series is the LINESTRING
# you're trying to create below :)
###############

all_coords = mapping(geom[0])["coordinates"]
##Convert coordinates to line string (May be a better way but allows usage of shapely)
shore_line = LineString(all_coords)

###############
print("\nComparing shore_line and shape_file.geometry.iat[0] to see if they're the same LINESTRING:", shore_line == shape_file.geometry.iat[0])
# In other words, the above code can be replaced with the following line: 
shore_line = shape_file.geometry.iat[0]
###############


##Define the sampling distance along the shoreline (should be >= the raster resolution)
distance_delta = 100

###############
# The builtin range function could save memory here for larger ranges (unless you need fractional values):
distances = range(0, int(shore_line.length), distance_delta)
###############
distances = np.arange(0, shore_line.length, distance_delta)


##Generates points along the shoreline
points = [shore_line.interpolate(distance) for distance in distances]+[shore_line.boundary[1]]
multipoints = unary_union(points)

xs = [multipoint.x for multipoint in multipoints]
ys = [multipoint.y for multipoint in multipoints]

xa = np.array(xs)
ya = np.array(ys)

##Calculates angle betwen point 1 and 3 **Needs to be updated to iterate through all points
angle= (math.radians(90)-(math.atan2((ya[2:3]-ya[0:1]),(xa[2:3]-xa[0:1]))))

## 90 degrees from the shore towards the land
a1 = angle + math.radians(90)
## 90 degrees from the shore towards the water
a2 = angle - math.radians(90)

##This creates the transect from the origin of a point along the shoreline, at an angle perpendicular to the shore,// and for a given length this example is 200 m
x1= xa[1:2]+(math.sin(a1)*200)
x2= xa[1:2]+(math.sin(a2)*200)

y1= ya[1:2]+(math.cos(a1)*200)
y2= ya[1:2]+(math.cos(a2)*200)

transect = LineString(([x1,y1],[x2,y2]))


##Create plots
plt.plot(*shore_line.xy, color='k')
plt.scatter(xs,ys, color='k')
plt.plot(*transect.xy, color='r', linewidth = 1.5, linestyle = ':')

plt.show()


###############
# Generating LineStrings for each transect:

# Function to generate the LineString for a given transect / row in the dataframe (see below first)
def get_transect(xy_row):
    if xy_row.hasnans:
        return None
    else:
        return LineString(((xy_row["landx"], xy_row["landy"]), (xy_row["waterx"], xy_row["watery"])))

# Here I create a DataFrame for the x and y values. The x column is almost 
# equivalent to your xa variable, as is the y column to your ya variable.
xy_data = pd.DataFrame({"x" : xs, "y" : ys})

# Constant value for 90 degrees in radians
rad90 = math.radians(90)

# This line generates a new column in the DataFrame for the angle value for 
# each transect, except all of the angles are calculated at the same time.shape_file

# Keep in mind that something like xy_data["y"] - xy_data["x"] will go row by 
# row and subtract each pair of y and x values since they're columns of numbers. 
# Numpy allows us to do these kind of row-wise operations using functions like 
# arctan, sin, and cos (used below).

# xy_data["y"].shift(-1) will shift the rows of the column up 1 position, this
# way the first row contains values for the second transect, the second row
# contains values for the third transect, etc.
xy_data["angles"] = rad90 - np.arctan2(xy_data["y"].shift(-1) - xy_data["y"].shift(1), xy_data["x"].shift(-1) - xy_data["x"].shift(1))

# Calculate your x1 and x2 values for each row / transect
xy_data["landx"] = xy_data["x"] + (200 * np.sin(xy_data["angles"] + rad90))
xy_data["waterx"] = xy_data["x"] + (200 * np.sin(xy_data["angles"] - rad90))

# Calculate your y1 and y2 values for each row / transect
xy_data["landy"] = xy_data["y"] + (200 * np.cos(xy_data["angles"] + rad90))
xy_data["watery"] = xy_data["y"] + (200 * np.cos(xy_data["angles"] - rad90))

# Generate the LineString for each row / transect. "apply" performs a custom
# function (the get_transect one above in this case) on each row of the dataframe.
transects = xy_data.apply(get_transect, axis=1)

# Generate a GeoDataFrame for the transects
new_geometry = gpd.GeoDataFrame(crs=shape_file.geometry.crs, geometry=transects)

# Plotting
new_geometry.plot()
plt.plot(*shore_line.xy, color='k')
plt.show()
###############