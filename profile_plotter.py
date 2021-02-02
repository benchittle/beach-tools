#-------------------------------------------------------------------------------
# Name:        profile_plotter.py
# Version:     Python 3.9.1, pandas 1.2.0, numpy 1.19.3
# Authors:     Ben Chittle
#-------------------------------------------------------------------------------

import os, re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

BEN_IN = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\sample_data"
BEN_OUT = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\out.xlsx"
UNI_IN = r"E:\SA\Runs\Poly\tables"
UNI_OUT =  r"E:\SA\Runs\Poly\tables\b_poly.xlsx"


#############################################################
# Change these variables to modify the input and output paths
# (type the path using the format above if needed).
current_input = BEN_IN
current_output = BEN_OUT
#############################################################

### DUPLICATE
def read_mask_csvs(path_to_dir):
    """
    Reads all .csv files in the given directory and concatenates them into a
    single DataFrame. Each .csv file name should end with a number specifying
    the segment its data corresponds to (i.e. 'data_file19.csv' would be
    interpreted to contain data for segment 19).
    """
    # Default value since all testing data is from one state.
    STATE = np.uint8(29)
    INPUT_COLUMNS = ["LINE_ID", "FIRST_DIST", "FIRST_Z"]
    OUTPUT_COLUMNS = ["profile", "x", "y"]
    DTYPES = dict(zip(INPUT_COLUMNS, [np.uint16, np.float32, np.float32]))

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
                segment = np.int16(segment.group())

                # Read only the necessary columns and reorder them in the
                # DataFrame.
                csv_data = pd.read_csv(path_to_dir + file_name,
                                       usecols=INPUT_COLUMNS,
                                       dtype=DTYPES)[INPUT_COLUMNS]
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


### DUPLICATE
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


### DUPLICATE
def identify_crest(profile_xy, shore_x):
    """Returns the x coordinate of the dune crest."""
    y = profile_xy.loc[shore_x:]["y"]
            # Current y value is the largest so far
    filt = ((y > y.shift(1).expanding(min_periods=1).max())
            # Difference between current y value and minimum of next 20 > 0.6
            & (y - y.rolling(20).min().shift(-20) > 0.6)
            # Current y value > next 10
            & (y > y.rolling(10).max().shift(-10)))

    x_coord = filt.idxmax()
    if x_coord == filt.index[0]:
        return None
    else:
        return x_coord


### DUPLICATE
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


### DUPLICATE
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


### DUPLICATE
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


### identifier has to be [state, seg, prof]
def plot_profile(profile_xy, identifier, points=[[], []]):
    """Plot a profile given its xy data"""
    x = profile_xy["x"]
    y = profile_xy["y"]

    # If no figure exists, create a new one. Otherwise, get the current figure.
    if len(plt.get_fignums()) == 0:
        fig = plt.figure(figsize=(10, 10), facecolor="#666666")
    else:
        fig = plt.gcf()

    num_axes = len(fig.axes)
    # If 4 profiles have been plotted, use a new figure.
    if num_axes == 4:
        num_axes = 0
        fig = plt.figure(figsize=(10, 10), facecolor="grey")

    # Add axes to the current figure.
    ax = fig.add_subplot(4, 1, num_axes + 1, anchor="N", facecolor="#444444",
                         xmargin=0, ylim=(0, 12.5))
    ax.set_title("State: {}, Segment: {}, Profile: {}".format(*identifier))
    ax.set_aspect(10)
    ax.fill_between(x, y, color="tan")
    ax.plot(x, y, color="orange", linestyle="-")
    ax.plot(points[0], points[1], color="red", marker="o", linestyle="",
            markersize=4)
    plt.tight_layout(h_pad=0.1)


### ASSUMES DATA CAN BE GROUPED BY state, segment, profile
### ASSUMES identifier IS OF FORM [num1, num2, num3]
def main(input_path, output_path):
    print("\nReading .csvs...")
    xy_data = read_mask_csvs(input_path)
    #print("Reading profiles...")
    #profiles = read_profiles(output_path)

    help_msg = ("\nPlotting Data:"
        "\n- Enter the 3 number identifier of a profile, seperated by any"
        " characters, to prepare a plot (i.e. '29 18 0')"
        "\n- Type 'plot' to display the data"
        "\n- Press ctrl+C or type 'quit' or 'q' to stop the script"
        "\n- Type 'help' to display this message\n")

    print(help_msg)
    grouped_xy = xy_data.groupby(["state", "segment", "profile"])
    identifiers = xy_data.set_index(["state", "segment", "profile"]).index
    while True:
        action = input("> ").lower().strip()
        if action == "plot":
            print("Plotting data")
            plt.show()
        elif action in ("q", "quit"):
            break
        elif action == "help":
            print(help_msg)
        else:
            identifier = tuple([int(i) for i in re.findall(r"\d+", action)])
            if len(identifier) == 3:
                if identifier in identifiers:
                    feature_data = identify_features(grouped_xy.get_group(identifier))
                    features_x = feature_data[::2]
                    features_y = feature_data[1::2]
                    plot_profile(grouped_xy.get_group(identifier),
                                 identifier=identifier,
                                 points=(features_x, features_y))

                    print("Profile {} ready".format(identifier))
                else:
                    print("The profile {} was not found in the data.".format(identifier))


if __name__ == "__main__":
    main(current_input, current_output)





