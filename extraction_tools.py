#-------------------------------------------------------------------------------
# Name:        extraction_tools.py
# Version:     Python 3.9.1, pandas 1.2.0, numpy 1.19.3
# Authors:     Ben Chittle, Alex Smith, Libby George
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np


############################### SHORE FUNCTIONS ###############################


def identify_shore_standard(xy_data, columns):
    grouped = xy_data.groupby(["state", "segment", "profile"])
    # Determine the slope of consecutive points over the data.
    #slope = xy_data["y"].diff(-1) / xy_data["x"].diff(-1)
    slope = grouped["y"].diff(1) / grouped["x"].diff(1)
    # Replace the last slope value for each profile with NaN (since this point
    # wouldn't have a slope value in reality, but because the data is stacked
    # together, it would use the first row in the next profile during the
    # above calculation).
    #slope.loc[xy_data["profile"] != xy_data["profile"].shift(-1)] = None

    # Filtering the data:
    return xy_data[
        # Minimum distance of 10.
        (xy_data["x"] > 10)
        # Minimum elevation of more than 0.75.
        & (xy_data["y"] > 0.75)
        # Maximum elevation of less than 0.85.
        & (xy_data["y"] < 0.85)
        # Positive or 0 slope in the next 2 values.
        & (slope.rolling(2).min().shift(-2) >= 0)
    # Extract the first position that satisfied the above conditions from each
    # profile, if any exist, and return the data for the selected columns.
    ].groupby(["state", "segment", "profile"]).head(1)[columns]
    
    '''xy_data.query(
        "(x > 10)"
        "& (y > 0.75)"
        "& (y < 0.85)"
        "& (@slope.rolling(2).min().shift(-2) >= 0)")[columns]'''


################################ TOE FUNCTIONS ################################


def identify_toe_rr(xy_data, shore_x, crest_x, columns):
    xy_data = xy_data.set_index(["state", "segment", "profile"])
    # Insert columns for the shore and crest values for each profile in xy_data.
    xy_data["shore_x"] = shore_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)
    xy_data["crest_x"] = crest_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)

    grouped = xy_data.groupby(["state", "segment", "profile"])
    # Filtering the data:
    return xy_data[
        # Toe must be past shore.
        (xy_data["x"] > xy_data["shore_x"])
        # Toe must be more than 3 units before crest.
        & (xy_data["x"] < xy_data["crest_x"] - 3)
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
    ].groupby(["state", "segment", "profile"]).head(1).reset_index()[columns]


def identify_toe_rrfar(xy_data, shore_x, crest_x, columns):
    xy_data = xy_data.set_index(["state", "segment", "profile"])
    # Insert columns for the shore and crest values for each profile in xy_data.
    xy_data["shore_x"] = shore_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)
    xy_data["crest_x"] = crest_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)

    grouped = xy_data.groupby(["state", "segment", "profile"])
    # Filtering the data:
    return xy_data[
        # Toe must be past shore.
        (xy_data["x"] > xy_data["shore_x"])
        # Toe must be more than 10 units before crest.
        & (xy_data["x"] < xy_data["crest_x"] - 10)
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
    ].groupby(["state", "segment", "profile"]).head(1).reset_index()[columns]


def identify_toe_ip(xy_data, shore_x, crest_x, columns):
    # Filtering the data:
    xy_data = xy_data[
        # Minimum distance of more than 45.
        (xy_data["x"] > 45)
        # Maximum distance of less than 63.
        & (xy_data["x"] < 63)
        # Maximum elevation of less than 3.2.
        & xy_data["y"] < 3.2]
    
    xy_data.set_index(["state", "segment", "profile"], inplace=True)
    xy_data.set_index("x", drop=False, append=True, inplace=True)

    grouped = xy_data.groupby(["state", "segment", "profile"])
    # Calculate the change in slope between each point for each profile.
    slope_change = (grouped["y"].diff(-1) / grouped["x"].diff(-1)).groupby(["state", "segment", "profile"]).diff(-1)

    # Identify the toe as the point with the greatest change in slope for each
    # profile and return the data for the selected columns.
    return xy_data.loc[slope_change.groupby(["state", "segment", "profile"]).idxmax()
        ].reset_index("x", drop=True).reset_index()[columns]
   

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
                # Toe must be at least 10 meters away from the crest. 10m has been specified for Libbys Site at Brackley Beach.
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
    xy_data = xy_data.set_index(["state", "segment", "profile"])
    # Insert columns for the shore and crest values for each profile in xy_data.
    xy_data["shore_x"] = shore_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)
    xy_data["crest_x"] = crest_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)

    # Apply the old polynomial toe identification function profile-wise on the 
    # DataFrame.
    toes = xy_data.dropna().groupby(["state", "segment", "profile"]).apply(_identify_toe_poly_old).rename("x")
    # Extract the rows corresponding to each identified toe in the original data.
    toes = xy_data.set_index("x", append=True).loc[pd.MultiIndex.from_frame(toe_x.reset_index())]

    # Return the data for the selected columns.
    return toes.reset_index()[columns]


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
    xy_data = xy_data.set_index(["state", "segment", "profile"])
    # Insert columns for the shore and crest values for each profile in xy_data.
    xy_data["shore_x"] = shore_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)
    xy_data["crest_x"] = crest_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)

    # Apply the old LCP toe identification function profile-wise on the 
    # DataFrame.
    toes = xy_data.dropna().groupby(["state", "segment", "profile"]).apply(_identify_toe_lcp_old).rename("x")
    # Extract the rows corresponding to each identified toe in the original data.
    toes = xy_data.set_index("x", append=True).loc[pd.MultiIndex.from_frame(toes.reset_index())]

    # Return the data for the selected columns.
    return toes.reset_index()[columns]


############################### CREST FUNCTIONS ###############################


def identify_crest_rr(xy_data, shore_x, columns):
    xy_data = xy_data.set_index(["state", "segment", "profile"])
    # Insert columns for the crest values for each profile in xy_data.
    xy_data["shore_x"] = shore_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)

    grouped = xy_data.groupby(["state", "segment", "profile"])
    return xy_data[
        # Crest must be past shore.
        (xy_data["x"] > xy_data["shore_x"])
        # Minimum distance of more than 30.
        & (xy_data["x"] > 30)
        # Maximum distance of less than 85.
        & (xy_data["x"] < 85)
        # Minimum relative relief of more than 0.55.
        & (xy_data["rr"] > 0.55)
#############################################################
#############################################################
        & (xy_data["y"] > grouped["y"].shift(-15)) 
#############################################################
#############################################################
        & (xy_data["rr"] > grouped["rr"].shift(2))
        & (xy_data["rr"] > grouped["rr"].shift(-2))
    ].groupby(["state", "segment", "profile"]).head(1).reset_index()[columns]


def identify_crest_standard(xy_data, shore_x, columns):
    xy_data = xy_data.set_index(["state", "segment", "profile"])
    xy_data["shore_x"] = shore_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)

    grouped = xy_data.groupby(["state", "segment", "profile"])
    return xy_data[
        (xy_data["x"] > xy_data["shore_x"])
        & (xy_data["y"] > grouped["y"].shift(1).groupby(["state", "segment", "profile"]).expanding(min_periods=1).max())
        & (xy_data["y"] - grouped["y"].rolling(20).min().groupby(["state", "segment", "profile"]).shift(-20) > 2)
        & (xy_data["y"] > grouped["y"].rolling(10).max().groupby(["state", "segment", "profile"]).shift(-10))
    ].groupby(["state", "segment", "profile"]).head(1).reset_index()[columns]


############################### CREST FUNCTIONS ###############################


def identify_heel_rr(xy_data, crest_x, columns):
    xy_data = xy_data.set_index(["state", "segment", "profile"])
    xy_data["crest_x"] = crest_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)
    xy_data.set_index("x", drop=False, append=True, inplace=True)

    grouped = xy_data.groupby(["state", "segment", "profile"])

    return xy_data.loc[xy_data[
        (xy_data["x"] > xy_data["crest_x"] + 5)
        & (grouped["rr"].shift(2) > 0.4)
        & (grouped["rr"].shift(-2) < 0.4)
    ].groupby(["state", "segment", "profile"])["y"].idxmin()].reset_index("x", drop=True).reset_index()[columns]


def identify_heel_standard(xy_data, crest_x, columns):
    xy_data = xy_data.set_index(["state", "segment", "profile"])
    xy_data["crest_x"] = crest_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)

    grouped = xy_data.groupby(["state", "segment", "profile"])

    mask = ((xy_data["x"] > xy_data["crest_x"])
        &  ((xy_data["y"] - grouped["y"].rolling(10).min().groupby(["state", "segment", "profile"]).shift(-10) <= 0.6)
            | (xy_data["y"] <= grouped["y"].rolling(10).max())
            | (xy_data["y"] <= grouped["y"].rolling(10).max().groupby(["state", "segment", "profile"]).shift(-10))))

    xy_data.set_index("x", drop=False, append=True, inplace=True)
    mask.index = xy_data.index
    return xy_data.loc[xy_data[mask].groupby(["state", "segment", "profile"])["y"].idxmin()].reset_index("x", drop=True).reset_index()[columns]


############################# PROFILE EXTRACTION ##############################
MODES = {
    "rr" : {"shore":identify_shore_standard, "toe":identify_toe_rr, "crest":identify_crest_rr, "heel":identify_heel_rr},
    "rrfar" : {"shore":identify_shore_standard, "toe":identify_toe_rrfar, "crest":identify_crest_rr, "heel":identify_heel_rr},
    "ip" : {"shore":identify_shore_standard, "toe":identify_toe_ip, "crest":identify_crest_standard, "heel":identify_heel_standard},
    "poly" : {"shore":identify_shore_standard, "toe":identify_toe_poly, "crest":identify_crest_standard, "heel":identify_heel_standard},
    "lcp" : {"shore":identify_shore_standard, "toe":identify_toe_lcp, "crest":identify_crest_standard, "heel":identify_heel_standard}
}

def find_closest_x(xy_data, x):
    # x must have index of date state seg profile
    xy_data = xy_data.set_index(["state", "segment", "profile"])
    xy_data["dist"] = (xy_data["x"] - x.reset_index("date", drop=True).reindex_like(xy_data)).abs()
    xy_data.set_index("x", append=True, drop=False, inplace=True)
    new = xy_data.loc[xy_data.groupby(["state", "segment", "profile"])["dist"].idxmin().dropna()]
    new = new[new["dist"] < 1]
    new.drop(columns="dist", inplace=True)
    new.reset_index("x", drop=True, inplace=True)
    new.reset_index(inplace=True)
    return new


def identify_features(mode, xy_data, use_shorex=None, use_toex=None, use_crestx=None, use_heelx=None):
    columns = ["date", "state", "segment", "profile", "x", "y", "rr"]

    if use_shorex is None:
        shore = mode["shore"](xy_data, columns=columns)
    else:
        shore = find_closest_x(xy_data, use_shorex)
    if use_crestx is None:
        crest = mode["crest"](xy_data, shore_x=shore, columns=columns)
    else:
        crest = find_closest_x(xy_data, use_crestx)
    if use_toex is None:
        toe = mode["toe"](xy_data, shore_x=shore, crest_x=crest, columns=columns)
    else:
        toe = find_closest_x(xy_data, use_toex)
    if use_heelx is None:
        heel = mode["heel"](xy_data, crest_x=crest, columns=columns)
    else:
        heel = find_closest_x(xy_data, use_heelx)

    for data, name in zip((shore, toe, crest, heel), ("shore", "toe", "crest", "heel")):
        data.set_index(["date", "state", "segment", "profile"], inplace=True)
        data.rename(columns={"x" : name + "_x", "y" : name + "_y", "rr" : name + "_rr"}, inplace=True)

    return pd.concat([shore, toe, crest, heel], axis=1)