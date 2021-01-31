#-------------------------------------------------------------------------------
# Name:        extraction_tools.py
# Version:     Python 3.9.1, pandas 1.2.0, numpy 1.19.3
# Authors:     Ben Chittle, Alex Smith, Libby George
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import extraction_tools as extract_old


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

    return xy_data[
        (xy_data["x"] > 10)
        & (xy_data["y"] > 0.75)
        & (xy_data["y"] < 0.85)
        & (slope.rolling(2).min().shift(-2) >= 0)
    ].groupby(["state", "segment", "profile"]).head(1)[columns]
    
    '''xy_data.query(
        "(x > 10)"
        "& (y > 0.75)"
        "& (y < 0.85)"
        "& (@slope.rolling(2).min().shift(-2) >= 0)")[columns]'''


################################ TOE FUNCTIONS ################################


def identify_toe_rr(xy_data, shore_x, crest_x, columns):
    xy_data = xy_data.set_index(["state", "segment", "profile"])
    xy_data["shore_x"] = shore_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)
    xy_data["crest_x"] = crest_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)

    grouped = xy_data.groupby(["state", "segment", "profile"])
    return xy_data[
        (xy_data["x"] > xy_data["shore_x"])
        & (xy_data["x"] < xy_data["crest_x"] - 3)
        & (xy_data["rr"] <= 0.25)
        & (grouped["rr"].shift(1) < 0.25)
        & (grouped["rr"].shift(-1) > 0.25)
        & (xy_data["x"] > 50)
    ].groupby(["state", "segment", "profile"]).head(1).reset_index()[columns]


def identify_toe_rrfar(xy_data, shore_x, crest_x, columns):
    xy_data = xy_data.set_index(["state", "segment", "profile"])
    xy_data["shore_x"] = shore_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)
    xy_data["crest_x"] = crest_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)

    grouped = xy_data.groupby(["state", "segment", "profile"])
    return xy_data[
        (xy_data["x"] > xy_data["shore_x"])
        & (xy_data["x"] < xy_data["crest_x"] - 10)
        & (xy_data["rr"] >= 0.25)
        & (grouped["rr"].shift(1) > 0.25)
        & (grouped["rr"].shift(-1) < 0.25)
        & (xy_data["x"] > 40)
    ].groupby(["state", "segment", "profile"]).head(1).reset_index()[columns]


def identify_toe_ip(xy_data, shore_x, crest_x, columns):
    xy_data = xy_data[
        (xy_data["x"] > 45)
        & (xy_data["x"] < 63)
        & xy_data["y"] < 3.2].set_index(["state", "segment", "profile"])
    #shore_x = shore_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)
    #crest_x = crest_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)  
    xy_data.set_index("x", drop=False, append=True, inplace=True)

    grouped = xy_data.groupby(["state", "segment", "profile"])
    ### IS ARCTAN NECESSARY?
    slope_change = np.arctan(grouped["y"].diff(-1) / grouped["x"].diff(-1)).groupby(["state", "segment", "profile"]).diff(-1)

    return xy_data.loc[slope_change.groupby(["state", "segment", "profile"]).idxmax()].reset_index("x", drop=True).reset_index()[columns]
   
'''
def identify_toe_poly(xy_data, shore_x, crest_x, columns):    
    xy_data = xy_data.set_index(["state", "segment", "profile"])
    shore_x = shore_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)
    crest_x = crest_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)

    xy_data = xy_data[
        (xy_data["x"] > shore_x)
        & (xy_data["x"] < crest_x)]
    
    coeffs = xy_data.groupby(["state", "segment", "profile"]).apply(lambda df: np.polyfit(x=df["x"], y=df["y"], deg=3))
    coeffs = pd.DataFrame(coeffs.to_list(), columns=["a", "b", "c", "d"], index=coeffs.index).reindex(xy_data.index)

    #print(coeffs)

    differences = (xy_data["y"]
                - coeffs["a"] * xy_data["x"] * xy_data["x"] * xy_data["x"]
                - coeffs["b"] * xy_data["x"] * xy_data["x"]
                - coeffs["c"] * xy_data["x"]
                - coeffs["d"])

    return xy_data[differences[(crest_x - xy_data["x"] > 5) & (xy_data["x"] > 40)].idxmin()#][columns]
'''

############################### CREST FUNCTIONS ###############################


def identify_crest_rr(xy_data, shore_x, columns):
    xy_data = xy_data.set_index(["state", "segment", "profile"])
    xy_data["shore_x"] = shore_x.set_index(["state", "segment", "profile"])["x"].reindex_like(xy_data)

    grouped = xy_data.groupby(["state", "segment", "profile"])
    return xy_data[
        (xy_data["x"] > xy_data["shore_x"])
        & (xy_data["x"] > 30)
        & (xy_data["x"] < 85)
        & (xy_data["rr"] > 0.55)
        & (xy_data["y"] > grouped["y"].shift(-15)) ######################
        & (xy_data["rr"] > grouped["rr"].shift(-2))
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

    print(xy_data.shape)

    mask = ((xy_data["x"] > xy_data["crest_x"])
        &  ((xy_data["y"] - grouped["y"].rolling(10).min().groupby(["state", "segment", "profile"]).shift(-10) > 0.6)
            | (xy_data["y"] > grouped["y"].rolling(10).max())
            | (xy_data["y"] > grouped["y"].rolling(10).max().groupby(["state", "segment", "profile"]).shift(-10))))

    xy_data.set_index("x", drop=False, append=True, inplace=True)
    mask.index = xy_data.index
    return xy_data.loc[xy_data[mask].groupby(["state", "segment", "profile"])["y"].idxmin()].reset_index("x", drop=True).reset_index()[columns]


############################# PROFILE EXTRACTION ##############################


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


def identify_features_rr(xy_data, use_shorex=None, use_toex=None, use_crestx=None, use_heelx=None):
    columns = ["date", "state", "segment", "profile", "x", "y", "rr"]
    if use_shorex is None:
        shore = identify_shore_standard(xy_data, columns=columns[:-1])
    else:
        shore = find_closest_x(xy_data, use_shorex)

    if use_crestx is None:
        crest = identify_crest_rr(xy_data, shore_x=shore, columns=columns)
    else:
        crest = find_closest_x(xy_data, use_crestx)

    if use_toex is None:
        toe = identify_toe_rr(xy_data, shore_x=shore, crest_x=crest, columns=columns)
    else:
        toe = find_closest_x(xy_data, use_toex)
    
    if use_heelx is None:
        heel = identify_heel_rr(xy_data, crest_x=crest, columns=columns)
    else:
        heel = find_closest_x(xy_data, use_heelx)

    for data in (shore, toe, crest, heel):
        data.set_index(["date", "state", "segment", "profile"], inplace=True)

    shore.rename(columns={"x":"shore_x", "y":"shore_y"}, inplace=True)
    toe.rename(columns={"x":"toe_x", "y":"toe_y", "rr":"toe_rr"}, inplace=True)
    crest.rename(columns={"x":"crest_x", "y":"crest_y", "rr":"crest_rr"}, inplace=True)
    heel.rename(columns={"x":"heel_x", "y":"heel_y", "rr":"heel_rr"}, inplace=True)

    return pd.concat([shore, toe, crest, heel], axis=1)


def identify_features_rrfar(xy_data, use_shorex=None, use_toex=None, use_crestx=None, use_heelx=None):
    columns = ["date", "state", "segment", "profile", "x", "y", "rr"]
    if use_shorex is None:
        shore = identify_shore_standard(xy_data, columns=columns[:-1])
    else:
        shore = find_closest_x(xy_data, use_shorex)

    if use_crestx is None:
        crest = identify_crest_rr(xy_data, shore_x=shore, columns=columns)
    else:
        crest = find_closest_x(xy_data, use_crestx)

    if use_toex is None:
        toe = identify_toe_rrfar(xy_data, shore_x=shore, crest_x=crest, columns=columns)
    else:
        toe = find_closest_x(xy_data, use_toex)
    
    if use_heelx is None:
        heel = identify_heel_rr(xy_data, crest_x=crest, columns=columns)
    else:
        heel = find_closest_x(xy_data, use_heelx)

    for data in (shore, toe, crest, heel):
        data.set_index(["date", "state", "segment", "profile"], inplace=True)

    shore.rename(columns={"x":"shore_x", "y":"shore_y"}, inplace=True)
    toe.rename(columns={"x":"toe_x", "y":"toe_y", "rr":"toe_rr"}, inplace=True)
    crest.rename(columns={"x":"crest_x", "y":"crest_y", "rr":"crest_rr"}, inplace=True)
    heel.rename(columns={"x":"heel_x", "y":"heel_y", "rr":"heel_rr"}, inplace=True)

    return pd.concat([shore, toe, crest, heel], axis=1)