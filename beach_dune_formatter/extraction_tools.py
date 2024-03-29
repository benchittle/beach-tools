#-------------------------------------------------------------------------------
# Name:        extraction_tools.py
# Version:     Python 3.9.1, pandas 1.2.4, numpy 1.20.3
# Authors:     Ben Chittle, Alex Smith, Libby George
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np


class ForwardIndexer(pd.api.indexers.BaseIndexer):
    def get_window_bounds(self, num_values: int = 0, min_periods = None, center = None, closed = None):
        if center:
            raise ValueError("Centering not supported")
        if closed is not None:
            raise ValueError("Closed ends not supported")
        
        offset = self.window_size

        end = np.arange(1 + offset, num_values + 1 + offset, dtype="int64")
        start = end - self.window_size

        end = np.clip(end, 0, num_values)
        start = np.clip(start, 0, num_values)

        return start, end

class BackwardIndexer(pd.api.indexers.BaseIndexer):
    def get_window_bounds(self, num_values: int = 0, min_periods = None, center = None, closed = None):
        if center:
            raise ValueError("Centering not supported")
        if closed is not None:
            raise ValueError("Closed ends not supported")
        
        offset = -1

        end = np.arange(1 + offset, num_values + 1 + offset, dtype="int64")
        start = end - self.window_size

        end = np.clip(end, 0, num_values)
        start = np.clip(start, 0, num_values)

        return start, end


############################### SHORE FUNCTIONS ###############################


def identify_shore_standard(xy_data, columns):
    grouped = xy_data.groupby(xy_data.index.names)
    # Determine the slope of consecutive points over the data.
    slope = (xy_data["y"] - grouped["y"].shift(1)) / (xy_data["x"] - grouped["x"].shift(1))

    # Filtering the data:
    return xy_data[columns][
        # Minimum distance of 10.
        (xy_data["x"] > 10)
        # Minimum elevation of more than 0.75.
        & (xy_data["y"] > 0.75)
        # Maximum elevation of less than 0.85.
        & (xy_data["y"] < 0.85)
        # Positive or 0 slope in the next 2 values.
        & (slope.rolling(ForwardIndexer(window_size=2)).min() >= 0)
    ].groupby(xy_data.index.names).head(1)


################################ TOE FUNCTIONS ################################


### USE .between(left, right, inclusive=False)

def identify_toe_rr(xy_data, shore_x, crest_x, columns):
    grouped = xy_data.groupby(xy_data.index.names)
    # Filtering the data:
    return xy_data[columns][
        # Toe must be past shore.
        (xy_data["x"] > shore_x)
        # Toe must be more than 3 units before crest.
        & (xy_data["x"] < crest_x - 3)
        # Maximum relative relief of 0.25.
        & (xy_data["rr"] <= 0.25)
        # Previous relative relief less than 0.25.
        & (grouped["rr"].shift(1) < 0.25)
        # Next relative relief greater than 0.25.
        & (grouped["rr"].shift(-1) > 0.25)
        # 25 for november, sept and july are at 40, 50 for July2020
        & (xy_data["x"] > 50)
    # Extract the first position that satisfied the above conditions from each
    # profile, if any exist, and return the data for the selected columns.
    ].groupby(xy_data.index.names).head(1)


def identify_toe_rrfar(xy_data, shore_x, crest_x, columns):
    grouped = xy_data.groupby(xy_data.index.names)
    # Filtering the data:
    return xy_data[columns][
        # Toe must be past shore.
        (xy_data["x"] > shore_x)
        # Toe must be more than 10 units before crest.
        & (xy_data["x"] < crest_x - 10)
        # Minimum relative relief of 0.25.
        & (xy_data["rr"] >= 0.25)
        # Previous relative relief greater than 0.25.
        & (grouped["rr"].shift(1) > 0.25)
        # Next relative relief less than 0.25.
        & (grouped["rr"].shift(-1) < 0.25)
        # 25 for november, sept and july are at 40.
        & (xy_data["x"] > 40)
    # Extract the first position that satisfied the above conditions from each
    # profile, if any exist, and return the data for the selected columns.
    ].groupby(xy_data.index.names).head(1)


def identify_toe_ip(xy_data, shore_x, crest_x, columns):
    # Filtering the data:
    filtered_xy = xy_data[
        # Minimum distance of more than 45.
        (xy_data["x"] > 45)
        # Maximum distance of less than 63.
        & (xy_data["x"] < 63)
        # Maximum elevation of less than 3.2.
        & (xy_data["y"] < 3.2)]
    grouped = filtered_xy.groupby(filtered_xy.index.names)

    # Calculate the change in slope between consecutive points:
    slope_change = (
        # Calculate slope between consecutive points.
        ((filtered_xy["y"] - grouped["y"].shift(1)) / (filtered_xy["x"] - grouped["x"].shift(1)))
        # Calculate first differences of the slope.
        .diff())

    # Identify the toe as the point with the greatest change in slope for each
    # profile and return the data for the selected columns.
    return filtered_xy[columns][
        slope_change == slope_change.groupby(slope_change.index.names).transform("max")
        ].groupby(filtered_xy.index.names).head(1)
   

def _identify_toe_poly_old(profile_xy):
    """Returns the x coordinate of the dune toe."""
    profile_xy.set_index("x", drop=False, inplace=True)
    shore_x = profile_xy["shore_x"].iat[0]
    crest_x = profile_xy["crest_x"].iat[0]
    # A subset of the profile's data from the identified shore to the 
    # identified crest. The toe can only be found within this subset.
    subset = profile_xy.loc[shore_x:crest_x]
    
    # Distance values.
    x = subset["x"]
    # Elevation values.
    y = subset["y"]

    # Polynomial coefficients.
    A, B, C, D = np.polyfit(x=x, y=y, deg=3)
    # Subtract the elevation values by the polynomial.
    differences = y - ((A * x * x * x) + (B * x * x) + (C * x) + D)
    # Create a Truth Series for filtering the differences by certain conditions. 
                # Toe must be more than 5m from crest.
    filtered = ((crest_x - x > 5)
                # Toe must be at least 40 meters away from the crest. 10m has been specified for Libbys Site at Brackley Beach.
                & (x > 40))
    # The toe x coordinate is identified as the x coordinate corresponding to
    # the minimum value in the above differences series after being filtered.
    try:
        x_coord = differences[filtered].idxmin()
    except ValueError:
        return None
    if x_coord == shore_x or x_coord == crest_x:
        return None
    else:
        return x_coord


def identify_toe_poly(xy_data, shore_x, crest_x, columns): 
    # Insert columns for the shore and crest values for each profile in xy_data.
    xy_data["shore_x"] = shore_x
    xy_data["crest_x"] = crest_x

    # Apply the old polynomial toe identification function profile-wise on the 
    # DataFrame.
    toe = xy_data.dropna().groupby(xy_data.index.names).apply(_identify_toe_poly_old).rename("x")
    # Extract the rows corresponding to each identified toe in the original data.
    toe = xy_data.set_index("x", append=True).loc[pd.MultiIndex.from_frame(toe.reset_index().dropna())]

    # Return the data for the selected columns.
    return toe.reset_index("x")[columns]


def _identify_toe_lcp_old(profile_xy):
    """Returns the dune toe coordinates for the given profile."""
    profile_xy.set_index("x", drop=False, inplace=True)
    shore_x = profile_xy["shore_x"].iat[0]
    crest_x = profile_xy["crest_x"].iat[0]
    # Create a subset of the profile's data from the identified shore
    # to the identified crest. The dune toe can only be found within
    # this subset.
    subset = profile_xy.loc[shore_x:crest_x]
    # Distance values
    x = subset["x"]
    # Elevation values
    y = subset["y"]

    # Determines coefficients for the linear polynomial.
    A, B = np.polyfit(x=[shore_x, crest_x], y=[y.min(), y.max()], deg=1)
    # Subtract the elevation values by the polynomial.
    differences = y - (A * x + B)

    # Create a Truth Series for filtering the differences by certain conditions. 
                # Toe must be more than 2 units away from crest.
    filtered = ((crest_x - x > 2)
                # Toe must be more than 5 units away from shore.
                & (x - shore_x > 5))
                
    # Identifies the x value at the minimum cubic-elevation difference
    # value. (This part is different compared to the other functions;
    # however, additional conditions can still be added in the same
    # way.)
    x_coord = differences[filtered].idxmin()
    if x_coord == shore_x or x_coord == crest_x:
        return None
    else:
        return x_coord


def identify_toe_lcp(xy_data, shore_x, crest_x, columns): 
    # Insert columns for the shore and crest values for each profile in xy_data.
    xy_data["shore_x"] = shore_x
    xy_data["crest_x"] = crest_x

    # Apply the old LCP toe identification function profile-wise on the 
    # DataFrame.
    toe = xy_data.dropna().groupby(xy_data.index.names).apply(_identify_toe_lcp_old).rename("x")
    # Extract the rows corresponding to each identified toe in the original data.
    toe = xy_data.set_index("x", append=True).loc[pd.MultiIndex.from_frame(toe.reset_index())]

    # Return the data for the selected columns.
    return toe.reset_index("x")[columns]


############################### CREST FUNCTIONS ###############################


def identify_crest_rr(xy_data, shore_x, columns):
    grouped = xy_data.groupby(xy_data.index.names)

    # Filtering the data:
    return xy_data[columns][
        # Crest must be past shore.
        (xy_data["x"] > shore_x)
        # Minimum distance of more than 30.
        & (xy_data["x"] > 30)
        # Maximum distance of less than 85.
        & (xy_data["x"] < 85)
        # Minimum relative relief of more than 0.55.
        & (xy_data["rr"] > 0.55)
        # Current elevation is greater than next 10.
        & (xy_data["y"] > grouped["y"].rolling(ForwardIndexer(window_size=10)).max()) 
        # Curent relative relief is greater than previous 2.
        & (xy_data["rr"] > grouped["rr"].rolling(BackwardIndexer(window_size=2)).max())
        # Current relative relief is greater than next 2.
        & (xy_data["rr"] > grouped["rr"].rolling(ForwardIndexer(window_size=2)).max())
    # Extract the first position that satisfied the above conditions from each
    # profile, if any exist, and return the data for the selected columns.
    ].groupby(xy_data.index.names).head(1)


def identify_crest_standard(xy_data, shore_x, columns):
    grouped = xy_data.groupby(xy_data.index.names)

    # Filtering the data:
    return xy_data[columns][
        # Crest must be past shore.
        (xy_data["x"] > shore_x)
        # Curent elevation is the largest so far.
        #use cummax?
        & (xy_data["y"] > grouped["y"].shift(1).groupby(xy_data.index.names).expanding(min_periods=1).max())
        # There is an elevation decrease of more than 2 in the next 20 values.
        & (xy_data["y"] - grouped["y"].rolling(ForwardIndexer(window_size=20)).min() > 2)
        # Current elevation is greater than next 10.
        & (xy_data["y"] > grouped["y"].rolling(ForwardIndexer(window_size=10)).max())
    # Extract the first position that satisfied the above conditions from each
    # profile, if any exist, and return the data for the selected columns.
    ].groupby(xy_data.index.names).head(1)


############################### CREST FUNCTIONS ###############################


def identify_heel_rr(xy_data, crest_x, columns):
    grouped = xy_data.groupby(xy_data.index.names)
    
    # Filtering the data:
    return xy_data[columns][
        # Heel must be more than 5 units past the crest.
        (xy_data["x"] > crest_x + 5)
        # Previous 2 relative relief values are greater than 0.4.
        & (grouped["rr"].rolling(BackwardIndexer(window_size=2)).min() > 0.4)
        # Next 2 relative relief values are less than 0.4.
        & (grouped["rr"].rolling(ForwardIndexer(window_size=2)).max() < 0.4)
        ].groupby(xy_data.index.names).head(1)


def identify_heel_standard(xy_data, crest_x, columns):
    grouped = xy_data.groupby(xy_data.index.names)
    
    # Filtering the data:
    xy_filtered = xy_data[
        # Heel must be past crest.
        (xy_data["x"] > crest_x)
        # The following conditions specify the data to be filtered OUT when satisfied:
        # ('~' inverts the conditions)
            # There is a decrease in elevation of more than 0.6 in the next 10 values.
        &  ~((xy_data["y"] - grouped["y"].rolling(ForwardIndexer(window_size=10)).min() > 0.6)
            # Current elevation is greater than previous 10. 
            & (xy_data["y"] > grouped["y"].rolling(BackwardIndexer(window_size=10)).max())
            # Current elevation is greater than next 10.
            & (xy_data["y"] > grouped["y"].rolling(ForwardIndexer(window_size=10)).max()))]

    # Apply the mask to the data and return the row with the minimum y value 
    # for each profile after filtering, for the selected columns.
    return xy_filtered[
        xy_filtered["y"] == xy_filtered["y"].groupby(xy_filtered.index.names).transform("min")
        ][columns].groupby(xy_filtered.index.names).head(1)


############################# PROFILE EXTRACTION ##############################
# Default extraction modes.
EXTRACTION_METHODS = {
    "rr" : {"shore":identify_shore_standard, "toe":identify_toe_rr, "crest":identify_crest_rr, "heel":identify_heel_rr},
    "rrfar" : {"shore":identify_shore_standard, "toe":identify_toe_rrfar, "crest":identify_crest_rr, "heel":identify_heel_rr},
    "ip" : {"shore":identify_shore_standard, "toe":identify_toe_ip, "crest":identify_crest_standard, "heel":identify_heel_standard},
    "poly" : {"shore":identify_shore_standard, "toe":identify_toe_poly, "crest":identify_crest_standard, "heel":identify_heel_standard},
    "lcp" : {"shore":identify_shore_standard, "toe":identify_toe_lcp, "crest":identify_crest_standard, "heel":identify_heel_standard}
}


def project_feature(new_xy_data, old_feature_x, tolerance=1) -> pd.DataFrame:
    """
    Project feature positions from one point in time onto another point in time
    given xy data for the new point in time. Uses the closest x coordinate in 
    new_xy_data to the feature's old x coordinate, within the specified tolerance.

    ARGUMENTS:
    new_xy_data: DataFrame
      Must contain x and y data for some of the same profiles as old_feature_x
    old_feature_x: Series
      Single x value for each profile marking the horizontal location of a 
      feature.
    tolerance: float
      Maximum distance between old_feature_x x value and the closest x found in 
      new_xy_data.
    """

    # Along each profile, determine the horizontal distance to the feature from 
    # each x coordinate.
    # .subtract is used to allow level=-1 to be specified (less fiddling with 
    # aligning the two pieces of data).
    dist = new_xy_data["x"].subtract(old_feature_x.reset_index(drop=True), level=-1).abs()
    # Identify the new feature coords. For each profile:
    return new_xy_data[
        # look for the point in the new data closest horizontally to the 
        # feature's position in the old data.
        (dist.groupby(dist.index.names).transform("min") == dist) 
        # make sure the distance to the point does not exceed the tolerance.
        & (dist < tolerance)
        ]


def identify_features(
    extract_methods, 
    xy_data, 
    columns,
    project_shorex=None, 
    project_toex=None, 
    project_crestx=None, 
    project_heelx=None    
) -> pd.DataFrame:

    # Identify the shoreline for each profile. If any of 'project' arguments
    # were given, that feature will be identified by projecting x coords from a
    # previous point in time onto the new xy data.
    if project_shorex is None:
        shore = extract_methods["shore"](xy_data, columns=columns)
    else:
        shore = project_feature(xy_data, project_shorex)
    shore_x_aligned = shore["x"].align(xy_data)[0]

    # Identify the crest for each profile.
    if project_crestx is None:
        crest = extract_methods["crest"](xy_data, shore_x=shore_x_aligned, columns=columns)
    else:
        crest = project_feature(xy_data, project_crestx)
    crest_x_aligned = crest["x"].align(xy_data)[0]

    # Identify the toe for each profile.
    if project_toex is None:
        toe = extract_methods["toe"](xy_data, shore_x=shore_x_aligned, crest_x=crest_x_aligned, columns=columns)
    else:
        toe = project_feature(xy_data, project_toex)

    # Identify the heel for each profile.
    if project_heelx is None:
        heel = extract_methods["heel"](xy_data, crest_x=crest_x_aligned, columns=columns)
    else:
        heel = project_feature(xy_data, project_heelx)

    # Rename the columns of each feature's DataFrame
    for data, name in zip((shore, toe, crest, heel), ("shore", "toe", "crest", "heel")):
        data.rename(columns={var : name + "_" + var for var in columns}, inplace=True)

    # Combine the feature DataFrames into a single DataFrame.
    return pd.concat([shore, toe, crest, heel], axis=1)


def measure_volume(xy_data, start_values, end_values, base_elevations):
    xy_data = xy_data.set_index(["date", "state", "segment", "profile"])
    start_values = start_values.reindex(xy_data.index)
    end_values = end_values.reindex(xy_data.index)
    base_elevations = base_elevations.reindex(xy_data.index)
    xy_data.loc[:, "y"] -= base_elevations
    new_data = xy_data[(xy_data["x"] >= start_values) & (xy_data["x"] <= end_values)]
    print(new_data.groupby(["date", "state", "segment", "profile"]).apply(lambda df: np.trapz(x=df["x"], y=df["y"])))
