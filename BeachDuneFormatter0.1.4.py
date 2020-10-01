#-------------------------------------------------------------------------------
# Name:        BeachDuneFormatter0.1.4.py
# Version:     0.1.4
#              Python 3.7, Pandas 0.24
#
# Purpose:     Processing beach profile data.
#
# Authors:     Ben Chittle, Alex Smith
#
# Last Edited: 23-09-2020
#
# Changed:
#   -Implemented point-by-point volume calculation
#   -Ignore "poorly conditioned polyfit" warning
#
# Todo:
#   -Optimize actual (point-by-point) volume calculation
#   -Revise for integration with a GUI / more interactive experience
#   -Implement metrics and conversions (look for an existing library?)
#   -Clarify code (add comments, try to make things as explicit as possible)
#   -Revise filtering (perhaps a profile with only one feature that is outside
#    of our bounds does not need to be completely filtered out?)
#
#
# Notes:
#   -The state column needs to be changed manually for different states (see the
#     __init__ method in ProfileFormatter for the variable to change it.
#-------------------------------------------------------------------------------

import os
import itertools
import time
import random, re #temp?

import pandas
import numpy as np
import warnings

from matplotlib import pyplot

warnings.simplefilter("ignore", np.RankWarning)

class ProfileFormatter:

    VALID_EXTENSIONS = [".csv"]

    def __init__(self, path):

        if os.path.isdir(path):
            self.path = path
        else:
            raise ValueError("Target directory does not exist\n{}".format(path))
## CHANGE STATE HERE
        self.state = 29
        self.COLUMN_HEADERS = [   # Make this an argument?
            "FIRST_DIST",
            "FIRST_Z",
            "LINE_ID",
            "SRC_NAME"
            ] # year
        self.NEW_COLUMN_HEADERS = [ # Make this an argument?
            "x",
            "y",
            "profile",
            "src_name",
            #"year",
            ]

        self.raw_df = pandas.DataFrame()
        self.profile_df = pandas.DataFrame()

        self._setup()


    def _setup(self): #read other formats?
        """
        Extracts values from the input data based on the provided conditional statements.

        (Statements are currently hard-coded - see the functions defined below)
        """

        def identify_shore(profile_df):
            """Returns the shore coordinates for the given profile."""
            x = profile_df["x"]
            y = profile_df["y"]
            m = (y.shift(1) - y) / (x.shift(1) - x)
            df = profile_df[
                (y > 0)
                & (y > y.shift(1).expanding(min_periods=1).max()) # Current y value is the largest so far
                & (m.rolling(5).min().shift(-4) >= 0) # Current value and next 4 slope values are positive
                ]
            return (df["x"].iat[0], df["y"].iat[0]) if len(df) > 0 else (np.nan, np.nan)

        def identify_toe(profile_df, shore_x, crest_x):
            """Returns the dune toe coordinates for the given profile."""
            if crest_x is np.nan:
                return (np.nan, np.nan)
            subset = profile_df.loc[shore_x:crest_x].iloc[1:-2]
            A, B, C, D = np.polyfit( # Polynomial coefficients
                x = subset.index,
                y = subset["y"],
                deg = 3
                )
            subset = subset.assign(
                diff=subset["y"] -((A * subset.index**3) + (B * subset.index**2) + (C * subset.index) + D)
                )
            toe_x = subset["diff"].idxmin()
            return (toe_x, subset.at[toe_x, "y"])

        def identify_crest(profile_df, shore_x):
            """Returns the dune crest coordinates for the given profile."""
            if shore_x is np.nan:
                return (np.nan, np.nan)
            subset = profile_df.loc[shore_x:].iloc[1:]
            y = subset["y"]
            df = subset[
                (y > y.shift(1).expanding(min_periods=1).max()) # Current y value is the largest so far
                & (y - y.rolling(20).min().shift(-20) > 0.6) # Difference between current y value and minimum of next 20 > 0.6
                & (y > y.rolling(10).max().shift(-10)) # Current y value > next 20
                ]
            return (df["x"].iat[0], df["y"].iat[0]) if len(df) > 0 else (np.nan, np.nan)

        def identify_heel(profile_df, crest_x):
            """Returns the dune heel coordinates for the given profile."""
            if crest_x is np.nan:
                return (np.nan, np.nan)
            subset = profile_df.loc[crest_x:].iloc[1:]
            y = subset["y"]
            df = subset[~(
                (y - y.rolling(10).min().shift(-10) > 0.6) # Difference between current y value and minimum of next 10 > 0.6
                & (y > y.rolling(10).max()) # Current y value > max of previous 10 y values
                & (y > y.rolling(20).max().shift(-20))
                )]
            return (df["y"].idxmin(), df["y"].min()) if len(df) > 0 else (np.nan, np.nan)


        ## TEMPORARY SOLUTION
        self.FEATURES = ["shore", "toe", "crest", "heel"] # Make this an argument?
        profile_columns = [
            "state",
            "segment",
            "profile",   #, "year"]
            "shore_x",
            "shore_y",
            "toe_x",
            "toe_y",
            "crest_x",
            "crest_y",
            "heel_x",
            "heel_y"
            ]

        # Read each .csv in the target directory and validate the column names
        # and file names.
        df_list = list()
        for csv_file, csv_name in self._validate_dir(self.path):
            df = pandas.read_csv(self.path + "\\" + csv_file)
            if not set(self.COLUMN_HEADERS).issubset(set(df.columns)):
                raise NameError("Column headers not recognized for {}. Column names should be {}.".format(csv_file, self.COLUMN_HEADERS))
            seg = csv_name[-2:]
            if len(seg) == 2 and seg.isnumeric():
                df_list.append(df[self.COLUMN_HEADERS].assign(segment=int(csv_name[-2:])))
            else:
                raise ValueError("Unable to identify segment number (make sure the name of the input file ends with a 2 digit number representing the segment)")

        # Concatenate all of the given files together into one DataFrame.
        self.raw_df = pandas.concat(df_list).drop(columns="SRC_NAME")

        # Adds a column containing the state identifier.
        self.raw_df = self.raw_df.assign(state=self.state)

        # Renames the column headers to better represent their values.
        self.raw_df = self.raw_df.rename(
            columns=dict(zip(self.COLUMN_HEADERS, self.NEW_COLUMN_HEADERS))
            )

        self.raw_df = self.raw_df.set_index(["state", "segment", "profile"]).sort_index(level=1)

        # A dictionary for storing the extracted data to be converted into a
        # DataFrame.
        profile_data = {header:list() for header in profile_columns}
        for segment, segment_df in self.raw_df.groupby("segment"):
            print("\nBEGINNING SEGMENT {}".format(segment))
            t1 = time.perf_counter()
            for profile, profile_df in segment_df.groupby("profile"):
                profile_data["state"].append(profile_df.index[0][0])
                profile_data["segment"].append(profile_df.index[0][1])
                profile_data["profile"].append(profile_df.index[0][2])

                profile_df.index = profile_df["x"]

                shore_coord = identify_shore(profile_df)
                profile_data["shore_x"].append(shore_coord[0])
                profile_data["shore_y"].append(shore_coord[1])

                crest_coord = identify_crest(profile_df, shore_coord[0])
                profile_data["crest_x"].append(crest_coord[0])
                profile_data["crest_y"].append(crest_coord[1])

                toe_coord = identify_toe(profile_df, shore_coord[0], crest_coord[0])
                profile_data["toe_x"].append(toe_coord[0])
                profile_data["toe_y"].append(toe_coord[1])

                heel_coord = identify_heel(profile_df, crest_coord[0])
                profile_data["heel_x"].append(heel_coord[0])
                profile_data["heel_y"].append(heel_coord[1])
            print("\tTIME: segment={}, t={}".format(segment, time.perf_counter() - t1))

        # Create a new dataframe using the data from the conditions
        self.profile_df = pandas.DataFrame(profile_data).set_index(["state", "segment", "profile"]).sort_index(level=[1, 2])


    def plot_data(self, loc=""):
        if loc == "":
            _, start_seg, start_prof = min(self.raw_df.index)
            _, end_seg, _ = max(self.raw_df.index)
            end_prof = int(np.mean(self.raw_df.index.codes[2]))
            print("segs: {}-{}, profs: {}-{}".format(start_seg, end_seg, start_prof, end_prof))
            loc = "{},{}".format(random.randint(start_seg, end_seg), random.randint(start_prof, end_prof))
            print("loc={}".format(loc))

        if re.compile("^[0-9]+,[0-9]+$").match(loc):
            print("Correct format")
            coord = tuple([self.state] + list(map(int, loc.split(","))))
            print("Coord: {}".format(coord))

            try:
                x_vals = self.raw_df.at[coord, "x"]
            except:
                print("Segment or profile out of range.")
            else:
                y_vals = self.raw_df.at[coord, "y"]
                #rr_vals = self.raw_df.at[coord, "rr"]

                feature_x = [self.profile_df.at[coord, feature + "_x"] for feature in self.FEATURES]
                feature_y = [self.profile_df.at[coord, feature + "_y"] for feature in self.FEATURES]

                fig, ax1 = pyplot.subplots(num="{}".format(coord))
                ax1.plot(x_vals, y_vals, "b-")
                #ax2 = ax1.twinx()
                #ax2.set_ylim([0, 1])
                #ax2.plot(x_vals, rr_vals, "r-")
                ax1.plot(feature_x, feature_y, "go")
                print("ready")
        else:
            print("Invalid input")


    def _validate_dir(self, path):
        """Check for necessary files in the target directory."""
        for file_name in os.listdir(path):
            name, extension = os.path.splitext(file_name)
            if extension in ProfileFormatter.VALID_EXTENSIONS:
                seg = name[-2:]
                if len(seg) == 2 and seg.isnumeric():
                    yield file_name, name
                else:
                    raise ValueError("Unable to identify segment number (make sure the name of the input file ends with a 2 digit number representing the segment)\nFile: {}".format(file_name))
            elif file_name == "XY.txt":
                pass
            else:
                print("Skipping file {}".format(file_name))


    def get_volume(self, start_feature, end_feature):
        """
        Determines the area of the profile between two key features.
        (Since the profiles are spaced at 1m, this can be interpreted as the volume.)
        """
        feature_volumes = []
        for segment, segment_data in self.raw_df.groupby("segment"):
            for profile, profile_data in segment_data.groupby("profile"):
                state = profile_data.index[0][0]
                data = profile_data[["x", "y"]].set_index("x")
                start_x, base_y, end_x = self.profile_df.loc[(state, segment, profile),
                                                       (start_feature + "_x", start_feature + "_y", end_feature + "_x")]
                feature_data = data.loc[start_x:end_x].reset_index()

                feature_volumes.append(sum(((feature_data["x"].shift(-1) - feature_data["x"]) * (feature_data["y"].shift(-1) - base_y)).dropna()))

        return feature_volumes



def pairwise(iterable):
    """
    Iterate pairwise through an iterable.

    pairwise([1,2,3,4]) -> (1,2),(2,3),(3,4)
    """
    val, nextVal = itertools.tee(iterable)
    next(nextVal, None)
    return zip(val, nextVal)


def _average_data(main_df):
    """Returns a new dataset averaged by every 10 profiles for each segment."""
    data = {header:list() for header in main_df}
    for segment, df in main_df.groupby(level=1):
        max_index = len(df)
        for header in df:
            data[header] += [df[header][index:index + 10].mean() for index in range(0, max_index, 10) if index + 10 < max_index]
    return data



def main():
    ## TEMP
    print("-Make sure to close the 'out.xlsx' document before the script tries writing to it, otherwise you will get an error\n")
    pandas.options.display.max_columns = 12 #temp
    pandas.options.display.width = 150 #temp

##  Specify the path to the input directory and output file.
    path = r"E:\SA\Runs\Poly\tables"
    excel_out = r"E:\SA\Runs\Poly\tables\b_poly.xlsx"

    all_data = ProfileFormatter(path)

    # Copies the DataFrame containing each feature's coordinates for all profiles.
    profiles = all_data.profile_df #"year"

    # Creates a new DataFrame to contain all of the unfiltered calculated
    # data for the features (e.g. beach width, dune height, etc.)
    print("\nCalculating feature data...")
    print("\tBasic features...")
    unfiltered_data = pandas.DataFrame(
        data={
            "dune_height" : profiles["crest_y"] - profiles["toe_y"],
            "beach_width" : profiles["toe_x"] - profiles["shore_x"],
            "dune_toe" : profiles["toe_y"],
            "dune_crest" : profiles["crest_y"],
            "dune_length" : profiles["crest_x"] - profiles["toe_x"]
            },
        index=profiles.index
        )
    unfiltered_data = unfiltered_data.assign(
        beach_slope=(profiles["toe_y"] - profiles["shore_y"]) / unfiltered_data["beach_width"],
        triangular_beach_vol=0.5 * unfiltered_data["beach_width"] * (profiles["toe_y"] - profiles["shore_y"]),
        triangular_dune_vol=0.5 * unfiltered_data["dune_length"] * unfiltered_data["dune_height"],
        dune_slope=unfiltered_data["dune_height"] / unfiltered_data["dune_length"],
        bd_ratio=(unfiltered_data["dune_height"] * unfiltered_data["dune_length"]) / ((profiles["toe_y"] - profiles["shore_y"]) * unfiltered_data["beach_width"])
        )

    # Calculates a more accurate volume value for the beach and dune by summing
    # the rectangular areas under the profile curve between consecutive points.
    print("\tActual volumes... (this may take a while)")
    unfiltered_data["actual_beach_vol"] = all_data.get_volume("shore", "toe")
    unfiltered_data["actual_dune_vol"] = all_data.get_volume("toe", "crest")

    # Creates a new DataFrame to contain all of the filterd calculated
    # data for the features.
    print("\tFiltering data...")
    filtered_data = unfiltered_data.dropna()[
          (unfiltered_data["triangular_dune_vol"] < 300)
        & (unfiltered_data["triangular_beach_vol"] < 500)
###        & (unfiltered_data["actual_dune_vol"] < 300)
###        & (unfiltered_data["actual_beach_vol"] < 500)
        & (unfiltered_data["dune_height"] > 1)
        & (unfiltered_data["dune_height"] < 10)
        & (unfiltered_data["dune_length"] < 25)
        & (unfiltered_data["dune_length"] > 5)
        & (unfiltered_data["dune_crest"] < 20)
        & (unfiltered_data["beach_width"] > 10)
        & (unfiltered_data["beach_width"] < 60)
        ]
    print("\nThere are {} profiles remaining after filtering out {} rows of"
          " data with values over our limits.".format(len(filtered_data), len(unfiltered_data)))

    # Creates a new DataFrame to contain all of the averaged calculated
    # data for the features. (Uses the unfiltered data and averages by
    # every 10 profiles for each segment)
    max_index = len(unfiltered_data)
    data_avg = pandas.DataFrame(
        data=_average_data(unfiltered_data)
        )
    print("\nThere are {} averages from {} rows of data.".format(len(data_avg), max_index))

    print("\nWriting to file... (this may take a while)")
    # Writes each DataFrame to a sheet in an Excel file.
    with pandas.ExcelWriter(excel_out) as writer:
        try:
            profiles.to_excel(writer, "profiles")
            print("\tDone getting profiles.")
            unfiltered_data.to_excel(writer, "data_nolimits")
            unfiltered_data.corr(method="pearson").to_excel(writer, "corr_1")
            filtered_data.to_excel(writer, "data_withlimits")
            filtered_data.corr(method="pearson").to_excel(writer, "corr_2")
            data_avg.to_excel(writer, 'avg')
        except IndexError:
            print("\nONE OR MORE DATAFRAMES WERE EMPTY AND COULDN'T BE WRITTEN"
                  " TO AN EXCEL FILE.")

    # Allow the user to input a segment and profile coordinate to plot a
    # profile.
    print("\nPlotting profiles:\n"
          "- <segment ID>,<profile ID> to select a profile (E.g. '2,104')\n"
          "- leave blank to try to select a random profile in an estimated range (if segment or profile is outside of range, keep trying)\n"
          "- 'plot' to plot selected data\n"
          "- 'quit' or ctrl+c to stop"
          )
    while True:
        print()
        action = input("Select a profile or plot selected profiles:\n").lower()
        if action == "plot":
            print("Plotting data...")
            pyplot.show()
        elif action == "quit":
            break
        else:
            all_data.plot_data(action)


    print("Done")

if __name__ == "__main__":
    main()



