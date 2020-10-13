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
A)


B)
Function to identify primary features for a given profile
1. START
2. READ (xy_data)
3. shore <- identify_shore(xy_data)
4. toe <- identify_toe(xy_data, shore)
5. crest <- identify_crest(xy_data, toe)
6. heel <- identify_heel(xy_data, crest)
7. RETURN (shore, toe, crest, heel)
8. END

Function to

C)


"""
import os, re, time
import pandas as pd
import numpy as np




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


### ARE ALL 8 NONES NEEDED?
def identify_features(profile_xy):
    """
    Identifies the coordinates of the shoreline, dune toe, dune crest, and
    dune heel for a given profile. These coordinates are returned as a tuple of
    8 components:
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
    start_x: Float
      The x coordinate of the start of the range.
    end_x: Float
      The x coordinate of the end of the range.
    profile_spacing: Float
      The distance between consecutive profiles.
    base_elevation: Float
      The elevation considered to be the ground / base. (Default is 0)
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
def measure_feature_volumes(grouped_xy_data, feature_data, start, end,
                            base_elevation, profile_spacing=None):
    """
    Returns a list of the volume measured between two points in each profile in
    the dataset.

    ARGUMENTS
    grouped_xy_data: DataFrame groupby object
      xy data for a set of profiles, grouped lastly by profile.
    feature_data: DataFrame
      Used to lookup the start and end coordinates for each profile.
    start: Column name
      Name of the column containing the start values in 'feature_data'.
    end: Column name
      Name of the column containing the end values in 'feature_data'.
    base_elevation: Column name or float
      If a column name is supplied, the values of this column in 'feature_data'
      will be used to measure volume from. If no column name is found and a
      numeric value is supplied, this value will be used as the base for all
      calculations.
    profile_spacing: Float
      The distance between consecutive profiles. If no value is supplied, the
      distance between the first two consecutive x values will be used (i.e. a
      square grid is assumed).

    """
    if profile_spacing is None:
        #
        first_group_x = grouped_xy_data.get_group(list(grouped_xy_data.groups)[0])["x"]
        profile_spacing = first_group_x.iat[1] - first_group_x.iat[0]

    data = []
    try:
        for index, profile_xy in grouped_xy_data:
            start_x, end_x, base_y = feature_data.loc[index, [start, end, base_elevation]]
            data.append(measure_volume(profile_xy, start_x, end_x, profile_spacing, base_y))
    except IndexError:
        for index, profile_xy in grouped_xy_data:
            start_x, end_x = feature_data.loc[index, [start, end]]
            data.append(measure_volume(profile_xy, start_x, end_x, profile_spacing, base_elevation))

    return data


def average_data(beach_data, size):
    return beach_data.rolling(size).mean().shift(1 - size).iloc[::size]


def test_1():
    BENPATH = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\sample_data"
    pd.options.display.max_columns = 20
    pd.options.display.width = 100

    xy_data = read_mask_csvs(BENPATH)
    #xy_data.set_index(["state", "segment", "profile"], inplace=True)
    print(xy_data)


def test_2():
    BENPATH = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\sample_data"
    BENOUT = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\out.xlsx"
    UNIPATH = r"E:\SA\Runs\Poly\tables"
    UNIOUT =  r"E:\SA\Runs\Poly\tables\b_poly.xlsx"

    current_input = BENPATH
    current_output = BENOUT

    FEATURE_COLUMNS = ["shore_x",  "shore_y", "toe_x", "toe_y", "crest_x",
                       "crest_y", "heel_x", "heel_y"]

    pd.options.display.max_columns = 12
    pd.options.display.width = 100

    print("\nReading .csv's...")
    xy_data = read_mask_csvs(current_input)

    print("\nIdentifying features...")
    t1 = time.perf_counter()

    # Identify the shoreline, dune toe, dune crest, and dune heel for each
    # profile in the data. This data will be returned as a Pandas Series
    # containing tuples of the 4 pairs of coordinates for each profile.
    profiles = xy_data.groupby(["state", "segment", "profile"]).apply(identify_features)

    # Expand the Series of tuples into a DataFrame where each column contains an
    # x or y componenent of a feature.
    profiles = pd.DataFrame(profiles.to_list(), columns=FEATURE_COLUMNS, index=profiles.index)
    print("\n\tTook {:.2f} seconds".format(time.perf_counter() - t1))

    print("\nCalculating beach data...")
    t1 = time.perf_counter()
    beach_data = pd.DataFrame(
        data={"dune_height" : profiles["crest_y"] - profiles["toe_y"],
              "beach_width" : profiles["toe_x"] - profiles["shore_x"],
              "dune_toe" : profiles["toe_y"],
              "dune_crest" : profiles["crest_y"],
              "dune_length" : profiles["crest_x"] - profiles["toe_x"]},
        index=profiles.index)
    beach_data["beach_slope"] = (profiles["toe_y"] - profiles["shore_y"]) / beach_data["beach_width"]
    beach_data["dune_slope"] = beach_data["dune_height"] / beach_data["dune_length"]
    beach_data["beach_vol"] = measure_feature_volumes(
                                     xy_data.groupby(["state", "segment", "profile"]),
                                     feature_data=profiles, start="shore_x",
                                     end="toe_x", base_elevation="shore_y")
    beach_data["dune_vol"] = measure_feature_volumes(
                                    xy_data.groupby(["state", "segment", "profile"]),
                                    feature_data=profiles, start="toe_x",
                                    end="crest_x", base_elevation="toe_y")
    ### SHOULD WE BE USING THE MORE ACCURATE VOLUMES FOR THE RATIO?
    beach_data["bd_ratio"] = beach_data["dune_vol"] / beach_data["beach_vol"]
    print("\n\tTook {:.2f} seconds".format(time.perf_counter() - t1))

    print("\nFiltering data...")
    t1 = time.perf_counter()
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
    print("\n\tTook {:.2f} seconds".format(time.perf_counter() - t1))
    print("\n\tThere are {} profiles remaining after filtering out {} rows of"
          " data.".format(len(filtered_beach_data), len(beach_data) - len(filtered_beach_data)))

    print("\nAveraging data...")
    t1 = time.perf_counter()
    averaged_beach_data = pd.concat(
        [average_data(df, 10) for index, df in beach_data.groupby(["state", "segment"])])
    print("\n\tTook {:.2f} seconds".format(time.perf_counter() - t1))

    print("\nWriting to file...")
    t1 = time.perf_counter()
    with pd.ExcelWriter(current_output) as writer:
        data_to_write = iter([("profiles", profiles),
                              ("unfiltered", beach_data),
                              ("corr_1", beach_data.corr(method="pearson")),
                              ("filtered", filtered_beach_data),
                              ("corr_2", filtered_beach_data.corr(method="pearson")),
                              ("averages", averaged_beach_data)])

        for name, data in data_to_write:
            try:
                data.to_excel(writer, name)
            except IndexError:
                print("\tFailed to write {} to file (the DataFrame may be"
                      " empty).".format(name))
    print("\n\tTook {:.2f} seconds".format(time.perf_counter() - t1))

    return xy_data, profiles, beach_data, filtered_beach_data

xy_data, profiles, beach_data, filtered_beach_data = test_2()


'''
def main():
    test_2()

if __name__ == "__main__":
    main()
'''

