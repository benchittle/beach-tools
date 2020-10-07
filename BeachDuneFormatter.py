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
import os, re
import pandas as pd



def read_mask_csvs(path_to_dir):
    """
    Reads all .csv files in the given directory and concatenates them into a
    single DataFrame. Each .csv file name should end with a number specifying
    the segment its data corresponds to.
    """
    # Default value since all testing data is from one state.
    STATE = 29
    INPUT_COLUMNS = ["LINE_ID", "FIRST_DIST", "FIRST_Z"]
    OUTPUT_COLUMNS = ["profile", "x", "y"]

    if not path_to_dir.endswith("\\"):
        path_to_dir += "\\"

    # Read each .csv file into a DataFrame and append it to a list.
    print("\nReading .csv's...")
    csvs = []
    for file_name in os.listdir(path_to_dir):
        head, extension = os.path.splitext(file_name)
        if extension == ".csv":
            # Look for a segment number at the end of the file name.
            segment = re.search(r"\d+$", head)

            # If a number was found, read the file and append it to the list.
            if segment is not None:
                segment = segment.group()

                # Read only the specified columns and reorder them in the
                # DataFrame.
                csv = pd.read_csv(path_to_dir + file_name,
                                  usecols=INPUT_COLUMNS)[INPUT_COLUMNS]
                csv.rename(columns=dict(zip(INPUT_COLUMNS, OUTPUT_COLUMNS)),
                           inplace=True)
                # Insert a segment and state column.
                csv.insert(loc=0, column="state", value=STATE)
                csv.insert(loc=1, column="segment", value=segment)
                csvs.append(csv)

                print("\tRead {} rows of data from '{}'".format(len(csv), file_name))
            else:
                print("\tSkipping '{}' (no segment number found)".format(file_name))
        else:
            print("\tSkipping '{}' (not a .csv)".format(file_name))

    return pd.concat(csvs)


def identify_features(profile_xy):
    """
    Identifies the coordinates of the shoreline, dune toe, dune crest, and
    dune heel for a given profile.
    """
    x = profile_xy["x"]
    y = profile_xy["y"]
    slope = (y - y.shift(1)) / (x - x.shift(1))

    ## IDENTIFY EACH FEATURE



def test_1():
    BENPATH = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\data"
    pd.options.display.max_columns = 12
    pd.options.display.width = 120

    xy_data = read_mask_csvs(BENPATH)
    #xy_data.set_index(["state", "segment", "profile"], inplace=True)
    print(xy_data)



def test_2():
    BENPATH = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\data"
    pd.options.display.max_columns = 12
    pd.options.display.width = 120

    xy_data = read_mask_csvs(BENPATH)
    print("\nStarting data manipulation")
    # Iterate over the data grouped first by state, then by segment, then by
    # profile.
    for (state, segment, profile), profile_xy in xy_data.groupby(["state", "segment", "profile"]):
        identify_features(profile_xy)
        break

def main():
    test_2()

if __name__ == "__main__":
    main()


