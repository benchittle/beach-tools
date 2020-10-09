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

    return pd.concat(csvs)


def identify_shore(profile_xy):
    """Returns a filter to identify the shoreline."""
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
    """Returns a filter to identify the dune crest."""
    y = profile_xy.loc[shore_x:]["y"]

    filt = (# Current y value is the largest so far
            (y > y.shift(1).expanding(min_periods=1).max())
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
    subset = profile_xy.loc[shore_x:crest_x]
    x = subset["x"]
    y = subset["y"]

    # Polynomial coefficients
    A, B, C, D = np.polyfit(x=x, y=y, deg=3)
    differences = y - ((A * x ** 3) + (B * x ** 2) + (C * x) + D)
    x_coord = differences.idxmin()
    return x_coord


def identify_heel(profile_df, crest_x):
    subset = profile_df.loc[crest_x:]
    y = subset["y"]

    filt = ~(# Difference between current y value and minimum of next 10 > 0.6
            (y - y.rolling(10).min().shift(-10) > 0.6)
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
    dune heel for a given profile.
    """
    #profile_xy = profile_xy.set_index("x", drop=False)

    shore_x = identify_shore(profile_xy)
    if shore_x is None:
        print("\tNo shore for {}".format(tuple(profile_xy.iloc[0]["state":"profile"])))
        return None, None, None, None, None, None, None, None

    crest_x = identify_crest(profile_xy, shore_x)
    if crest_x is None:
        print("\tNo crest for {}".format(tuple(profile_xy.iloc[0]["state":"profile"])))
        return None, None, None, None

    toe_x = identify_toe(profile_xy, shore_x, crest_x)
    if toe_x is None:
        print("\tNo toe for {}".format(tuple(profile_xy.iloc[0]["state":"profile"])))
        return None, None, None, None

    heel_x = identify_heel(profile_xy, crest_x)
    if heel_x is None:
        print("\tNo heel for {}".format(tuple(profile_xy.iloc[0]["state":"profile"])))
        return None, None, None, None

    shore_y, toe_y, crest_y, heel_y = profile_xy.loc[[shore_x, toe_x, crest_x, heel_x], "y"]

    return shore_x, shore_y, toe_x, toe_y, crest_x, crest_y, heel_x, heel_y


def test_1():
    BENPATH = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\data"
    pd.options.display.max_columns = 12
    pd.options.display.width = 120

    xy_data = read_mask_csvs(BENPATH)
    #xy_data.set_index(["state", "segment", "profile"], inplace=True)
    print(xy_data)


def test_2():
    BENPATH = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\data"
    UNIPATH = r"E:\SA\Runs\Poly\tables"
    FEATURE_COLUMNS =   ["shore_x",  "shore_y",
                       "toe_x", "toe_y", "crest_x", "crest_y", "heel_x",
                       "heel_y"]#["state", "segment", "profile",
    pd.options.display.max_columns = 12
    pd.options.display.width = 120

    print("\nReading .csv's...")
    xy_data = read_mask_csvs(BENPATH).set_index("x", drop=False)

    print("\nIdentifying features...")
    t1 = time.perf_counter()
    profiles = xy_data.groupby(["state", "segment", "profile"]).apply(identify_features)
    profiles = pd.DataFrame(profiles.to_list(), columns=FEATURE_COLUMNS, index=profiles.index)
    print("\n\tTook {}".format(time.perf_counter() - t1))

    print("\nCalculating beach data...")
    beach_data = pd.DataFrame(
        data={"dune_height" : profiles["crest_y"] - profiles["toe_y"],
              "beach_width" : profiles["toe_x"] - profiles["shore_x"],
              "dune_toe" : profiles["toe_y"],
              "dune_crest" : profiles["crest_y"],
              "dune_length" : profiles["crest_x"] - profiles["toe_x"]},
        index=profiles.index)
    beach_data["beach_slope"] = (profiles["toe_y"] - profiles["shore_y"]) / beach_data["beach_width"]
    beach_data["dune_slope"] = beach_data["dune_height"] / beach_data["dune_length"]
    beach_data["bd_ratio"] = ((beach_data["dune_height"] * beach_data["dune_length"])
                              / ((profiles["toe_y"] - profiles["shore_y"])
                              * beach_data["beach_width"]))

    print(beach_data.head())

def main():
    test_2()

if __name__ == "__main__":
    main()


