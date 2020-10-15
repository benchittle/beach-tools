#-------------------------------------------------------------------------------
# Name:        BeachDuneFormatter.py
# Version:     Python 3.7, Pandas 0.24
#
# Purpose:     Processing beach profile data.
#
# Authors:     Ben Chittle, Alex Smith
#-------------------------------------------------------------------------------

"""
3 parts:
    A) input checking and formatting
    B) data processing (create a library of functions)
    C) output (formatting)
"""
import os, re, time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


BEN_IN = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\sample_data"
BEN_OUT = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\out.xlsx"
UNI_IN = r"E:\SA\Runs\Poly\tables"
UNI_OUT =  r"E:\SA\Runs\Poly\tables\b_poly.xlsx"


####################### PATH SETTINGS #######################
# Change these variables to modify the input and output paths
# (type the path directly using the format above if needed).
current_input = BEN_IN
current_output = BEN_OUT
#############################################################


def read_mask_csvs(path_to_dir):
    """
    Reads all .csv files in the given directory and concatenates them into a
    single DataFrame. Each .csv file name should end with a number specifying
    the segment its data corresponds to (i.e. 'data_file19.csv' would be
    interpreted to contain data for segment 19).
    """
    # Default value since all testing data is from one state.
    STATE = 29
    INPUT_COLUMNS = ["LINE_ID", "FIRST_DIST", "FIRST_Z"]
    OUTPUT_COLUMNS = ["profile", "x", "y"]

    if not path_to_dir.endswith("\\"):
        path_to_dir += "\\"

    # Read each .csv file into a DataFrame and append the DataFrame to a list.
    csvs = []
    for file_name in os.listdir(path_to_dir):
        head, extension = os.path.splitext(file_name)
        if extension == ".csv":
            # Look for a segment number at the end of the file name.
            segment = re.search(r"\d+$", head)

            # If a number was found, read the file and append it to the list.
            if segment is not None:
                segment = int(segment.group())

                # Read only the necessary columns and reorder them in the
                # DataFrame.
                csv_data = pd.read_csv(path_to_dir + file_name,
                                  usecols=INPUT_COLUMNS)[INPUT_COLUMNS]
                csv_data.rename(columns=dict(zip(INPUT_COLUMNS, OUTPUT_COLUMNS)),
                           inplace=True)
                # Insert a column for the segment and state values.
                csv_data.insert(loc=0, column="state", value=STATE)
                csv_data.insert(loc=1, column="segment", value=segment)
                csvs.append(csv_data)

                print("\tRead {} rows of data from file '{}'".format(len(csv_data), file_name))
            else:
                print("\tSkipping file '{}' (no segment number found)".format(file_name))
        else:
            print("\tSkipping file '{}' (not a .csv)".format(file_name))

    return pd.concat(csvs).set_index("x", drop=False)


def identify_shore(profile_xy):
    """Returns the x coordinate of the shoreline."""
    x = profile_xy["x"]
    y = profile_xy["y"]
    slope = (y - y.shift(1)) / (x - x.shift(1))

    filt = ((y > 0)
            # Current y value is the largest so far
            & (y > y.shift(1).expanding(min_periods=1).max())
            # Current value and next 4 slope values are positive
            & (slope.rolling(5).min().shift(-4) >= 0))

    x_coord = filt.idxmax()
    if x_coord == filt.index[0]:
        return None
    else:
        return x_coord


def identify_crest(profile_xy, shore_x):
    """Returns the x coordinate of the dune crest."""
    y = profile_xy.loc[shore_x:]["y"]
            # Current y value is the largest so far
    filt = ((y > y.shift(1).expanding(min_periods=1).max())
            # Difference between current y value and minimum of next 20 > 0.6
            & (y - y.rolling(20).min().shift(-20) > 0.6)
            # Current y value > next 20
            & (y > y.rolling(10).max().shift(-10)))

    x_coord = filt.idxmax()
    if x_coord == filt.index[0]:
        return None
    else:
        return x_coord


### COMPARE TO NEW
def identify_toe_old(profile_xy, shore_x, crest_x):
    """Returns the x coordinate of the dune toe."""
    subset = profile_xy.loc[shore_x:crest_x].iloc[1:-2]
    x = subset["x"]
    y = subset["y"]

    # Polynomial coefficients
    A, B, C, D = np.polyfit(x=x, y=y, deg=3)
    differences = y - ((A * x ** 3) + (B * x ** 2) + (C * x) + D)
    x_coord = differences.idxmin()
    return x_coord


def identify_toe(profile_xy, shore_x, crest_x):
    """Returns the x coordinate of the dune toe."""
    subset = profile_xy.loc[shore_x:crest_x]
    x = subset["x"]
    y = subset["y"]

    # Polynomial coefficients
    A, B, C, D = np.polyfit(x=x, y=y, deg=3)
    differences = y - ((A * x ** 3) + (B * x ** 2) + (C * x) + D)
    x_coord = differences.idxmin()
    return x_coord


def identify_heel(profile_df, crest_x):
    """Returns the x coordinate of the dune heel."""
    subset = profile_df.loc[crest_x:]
    y = subset["y"]
             # Difference between current y value and minimum of next 10 > 0.6
    filt = ~((y - y.rolling(10).min().shift(-10) > 0.6)
             # Current y value > max of previous 10 y values
             & (y > y.rolling(10).max())
             & (y > y.rolling(20).max().shift(-20)))

    x_coord = y[filt].idxmin()
    if x_coord == filt.index[0]:
        return None
    else:
        return x_coord


def identify_features(profile_xy):
    """
    Returns the coordinates of the shoreline, dune toe, dune crest, and
    dune heel for a given profile as:
    (shore_x, shore_y, toe_x, toe_y, crest_x, crest_y, heel_x, heel_y)
    """
    shore_x = identify_shore(profile_xy)
    if shore_x is None:
        print("\tNo shore for {}".format(tuple(profile_xy.iloc[0]["state":"profile"])))
        return None

    crest_x = identify_crest(profile_xy, shore_x)
    if crest_x is None:
        print("\tNo crest for {}".format(tuple(profile_xy.iloc[0]["state":"profile"])))
        return None

    toe_x = identify_toe(profile_xy, shore_x, crest_x)
    if toe_x is None:
        print("\tNo toe for {}".format(tuple(profile_xy.iloc[0]["state":"profile"])))
        return None

    heel_x = identify_heel(profile_xy, crest_x)
    if heel_x is None:
        print("\tNo heel for {}".format(tuple(profile_xy.iloc[0]["state":"profile"])))
        return None

    shore_y, toe_y, crest_y, heel_y = profile_xy.loc[[shore_x, toe_x, crest_x, heel_x], "y"]

    return shore_x, shore_y, toe_x, toe_y, crest_x, crest_y, heel_x, heel_y


def measure_volume(profile_xy, start_x, end_x, profile_spacing, base_elevation=0):
    """
    Returns an approximation of the volume of the beach between two points.

    ARGUMENTS
    profile_xy: DataFrame
      xy data for a particular profile.
    start_x: float
      The x coordinate of the start of the range.
    end_x: float
      The x coordinate of the end of the range.
    profile_spacing: float
      The distance between consecutive profiles.
    base_elevation: float
      Set the height of the horizontal axis to measure volume from. Change this
      if y values are relative to an elevation other than y=0.
    """
    subset = profile_xy.loc[start_x:end_x]
    x = subset["x"]
    y = subset["y"]

    # Make all elevation values relative to the base elevation.
    y -= base_elevation

    # The area under the profile curve is calculated using the trapezoidal rule
    # and multiplized by the distance between consecutive profiles to
    # approximate the volume.
    return np.trapz(y=y, x=x) * profile_spacing


### MAKE THIS A HIGHER LEVEL / MULTIPLE PROFILES MODULE FUNCTION AND LEAVE
### get_volume AS A SINGLE PROFILE FUNCTION

###ASSUMES THAT DATA CAN BE GROUPED BY
def measure_feature_volumes(xy_data, start_values, end_values, base_elevations,
                            profile_spacing=None):
    """
    Returns a list of the volume measured between two points in each profile in
    the dataset.

    ARGUMENTS
    xy_data: DataFrame
      xy data of a collection of profiles.
    start_values: array like
      The x value to start measuring volume from for each profile.
    end_values: array like
      The x value to stop measuring volume from for each profile.
    base_elevations: array like
      The base elevation for each profile.
    profile_spacing: Float
      The distance between consecutive profiles. If no value is supplied, the
      distance between the first two consecutive x values will be used (i.e. a
      square grid is assumed).
    """
    if profile_spacing is None:
        profile_spacing = xy_data["x"].iat[1] - xy_data["x"].iat[0]

    grouped_xy = xy_data.groupby(["state", "segment", "profile"])
    data = []
    for (index, profile_xy), start_x, end_x, base_y in zip(grouped_xy, start_values, end_values, base_elevations):
        data.append(measure_volume(profile_xy, start_x, end_x, profile_spacing, base_y))

    return data


### ONLY EVERY nth VALUE NEEDS TO BE CALCULATED RATHER THAN CALCULATING THE
### ROLLING MEAN AND THEN TAKING EVERY nth VALUE
def average_data(beach_data, size):
    data = []
    for index, segment_beach_data in beach_data.groupby(["state", "segment"]):
        data.append(beach_data.rolling(size).mean().shift(1 - size).iloc[::size])
    return pd.concat(data)


def write_data_excel(path_to_file, dataframes, names):
    with pd.ExcelWriter(path_to_file) as writer:
        for name, data in zip(names, dataframes):
            try:
                data.to_excel(writer, name)
            except IndexError:
                print("\tFailed to write {} to file (the DataFrame may be"
                      " empty).".format(name))


### HAVE THE USER DECLARE HOW THEIR DATA IS CATEGORIZED / ORGANIZED
### (need to know for groupby operations)
def main(input_path, output_path):
    FEATURE_COLUMNS = ["shore_x",  "shore_y", "toe_x", "toe_y", "crest_x",
                       "crest_y", "heel_x", "heel_y"]

    pd.options.display.max_columns = 12
    pd.options.display.width = 100

    initial_start_time = time.perf_counter()

    print("\nReading .csv's...")
    start_time = time.perf_counter()
    xy_data = read_mask_csvs(input_path)
    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))


    print("\nIdentifying features...")
    start_time = time.perf_counter()

    # Identify the shoreline, dune toe, dune crest, and dune heel for each
    # profile in the data. This data will be returned as a Pandas Series
    # containing tuples of the 4 pairs of coordinates for each profile.
    profiles = xy_data.groupby(["state", "segment", "profile"]).apply(identify_features)

    # Expand the Series of tuples into a DataFrame where each column contains an
    # x or y componenent of a feature.
    profiles = pd.DataFrame(profiles.to_list(), columns=FEATURE_COLUMNS, index=profiles.index)
    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))


    print("\nCalculating beach data...")
    start_time = time.perf_counter()

    # Use the feature data for each profile to calculate additional
    # characteristics of the beach. This includes the dune height, beach width,
    # dune toe height, dune crest height, dune length, beach slope, dune slope,
    # beach volume, dune volume, and beach-dune volume ratio.
    beach_data = pd.DataFrame(
        data={"dune_height" : profiles["crest_y"] - profiles["toe_y"],
              "beach_width" : profiles["toe_x"] - profiles["shore_x"],
              "dune_toe" : profiles["toe_y"],
              "dune_crest" : profiles["crest_y"],
              "dune_length" : profiles["crest_x"] - profiles["toe_x"]},
        index=profiles.index)

    # Approximates the beach slope as the total change in height of the beach
    # divided by the length of the beach.
    beach_data["beach_slope"] = (profiles["toe_y"] - profiles["shore_y"]) / beach_data["beach_width"]

    # Approximates the dune slope as the change in height of the dune divided by
    # the length of the dune.
    beach_data["dune_slope"] = beach_data["dune_height"] / beach_data["dune_length"]

    beach_data["beach_vol"] = measure_feature_volumes(
                                  xy_data,
                                  start_values=profiles["shore_x"],
                                  end_values=profiles["toe_x"],
                                  base_elevations=profiles["shore_y"])
    ### ARE ELEVATIONS RELATIVE TO TOE Y OR SHORE Y?
    beach_data["dune_vol"] = measure_feature_volumes(
                                 xy_data,
                                 start_values=profiles["toe_x"],
                                 end_values=profiles["crest_x"],
                                 base_elevations=profiles["toe_y"])
    ### SHOULD THIS BE db_ratio?
    beach_data["bd_ratio"] = beach_data["dune_vol"] / beach_data["beach_vol"]
    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))


    print("\nFiltering data...")
    start_time = time.perf_counter()

    ### SHOULD THE ENTIRE ROW OF DATA BE WIPED OUT IF ONLY ONE FEATURE IS UNFIT?
    filtered_beach_data = beach_data[
          (beach_data["dune_vol"] < 300) ##
        & (beach_data["beach_vol"] < 500) ##
        & (beach_data["dune_height"] > 1)
        & (beach_data["dune_height"] < 10)
        & (beach_data["dune_length"] > 5)
        & (beach_data["dune_length"] < 25)
        & (beach_data["dune_crest"] < 20)
        & (beach_data["beach_width"] > 10)
        & (beach_data["beach_width"] < 60)].dropna()
    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))
    print("\tThere are {} profiles remaining after filtering out {} rows of"
          " data.".format(len(filtered_beach_data), len(beach_data) - len(filtered_beach_data)))


    print("\nAveraging data...")
    start_time = time.perf_counter()

    # Takes the mean of each column for every 10 profiles.
    averaged_beach_data = average_data(beach_data, size=10)
    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))


    print("\nCorrelating data...")
    start_time = time.perf_counter()
    corr1 = beach_data.corr(method="pearson")
    corr2 = filtered_beach_data.corr(method="pearson")
    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))


    print("\nWriting to file...")
    start_time = time.perf_counter()
    write_data_excel(path_to_file=output_path,
                     dataframes=(profiles, beach_data, corr1, filtered_beach_data,
                                 corr2, averaged_beach_data),
                     names=("profiles", "unfiltered", "corr_1", "filtered",
                            "corr_2", "averages"))
    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))

    print("\nTotal time: {:.2f} seconds".format(time.perf_counter() - initial_start_time))

    return xy_data, profiles, beach_data, filtered_beach_data, averaged_beach_data


if __name__ == "__main__":
    xy_data, profiles, beach_data, filtered_beach_data, avg = main(current_input, current_output)


