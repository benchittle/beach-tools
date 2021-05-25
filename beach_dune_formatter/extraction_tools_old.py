#-------------------------------------------------------------------------------
# Name:        extraction_tools.py
# Version:     Python 3.9.1, pandas 1.2.0, numpy 1.19.3
# Authors:     Ben Chittle, Alex Smith, Libby George
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np


############################### SHORE FUNCTIONS ###############################


def identify_shore_standard(profile_df):
    """Returns the shore coordinates for the given profile."""
    
    # Distance values
    x = profile_df["x"]
    # Elevation values
    y = profile_df["y"]
    # Slope values
    slope = (y.shift(1) - y) / (x.shift(1) - x)

    # shore_rr does not currently use the rr surface, therefore can be applied to any
    # of the other techniques below.
    
    # Create a Boolean mask for filtering the subset by certain conditions.
            # November > 10, Sept > 20
    mask = ((x > 10)
            # Current elevation is above 0.8m or MSL
            & (y > 0.75)
            # Current elevation is highest so far
            #& (y > y.shift(1).expanding(min_periods=1).max()) 
            # Current elevation is below 0.9m, to restrict it towards 0m or MSL
            & (y < 0.85)
            # Current  value < next value
            #& (y > y.rolling(1).min().shift(-1)) 
            # Positive slope (next 2 values)
            & (slope.rolling(2).min().shift(-2) >= 0))

    

    # The shore x coordinate is identified at the first position that satisfies
    # the filter. 
    x_index = mask.idxmax()
    # If no positions satistfied the filter, return None.
    if mask.at[x_index] == False:
        return None
    # Otherwise return the x_coordinate
    else:
        return x.at[x_index]


################################ TOE FUNCTIONS ################################


def identify_toe_rr(profile_df, shore_x, crest_x):
    """Returns the dune toe coordinates for the given profile."""
    # Create a subset of the profile's data from the identified shore to the 
    # identified crest. The dune toe can only be found within this subset.
    subset = profile_df.loc[shore_x:crest_x]
    # Distance values.
    x = subset["x"]
    # Relative relief values
    rr = subset["rr"]
    # Relative relief slope values
    #rr_slope = (rr.shift(1) - rr) / (x.shift(1) - x)

    # Create a Truth series for filtering the subset by certain conditions.
                # Distance is more than 3 meters from crest.
    filtered = ((crest_x - x > 3)
                # The toe must be greater than 5m from the shoreline 
                #& (x - shore_x > 5) 
                # Maximum relative relief of 0.25.
                & (rr <= 0.25)
                # Previous relative relief is less than 0.25.
                & (rr.shift(1) < 0.25)
                # Next relative relief is greater than 0.25.
                & (rr.shift(-1) > 0.25)
                # Positive slope (current and next 5 values)
                #& (rr_slope.rolling(5).min().shift(-5) >= 0)
                # Moved this to 25 for november, sept and july are at 40, 50 for July2020
                & (x > 50))
                
    # The toe x coordinate is identified at the first position that satisfies
    # the filter. 
    x_coord = filtered.idxmax()
    # If no positions satistfied the filter, return no toe.
    if filtered[x_coord] == False:
        return None
    else:
        return x_coord


def identify_toe_rrfar(profile_df, shore_x, crest_x):
    """Returns the dune toe coordinates for the given profile."""
    # Create a subset of the profile's data from the identified shore to the 
    # identified crest. The dune toe can only be found within this subset.
    subset = profile_df.loc[shore_x:crest_x]
    # Distance values.
    x = subset["x"]
    # Relative relief values
    rr = subset["rr"]
   
    # Relative relief slope values, this has not actually been used
    # before, is an idea for future tools / problems 
    #rr_slope = (rr.shift(1) - rr) / (x.shift(1) - x)

    # Create a Truth series for filtering the subset by certain conditions.
                # Distance is more than 10 meters from crest.
    filtered = ((crest_x - x > 10)
                # The toe must be greater than 5m from the shoreline 
                #& (x - shore_x > 5) 
                # Minimum relative relief of 0.25.
                & (rr >= 0.25)
                # Previous relative relief is greater than 0.25.
                & (rr.shift(1) > 0.25)
                # Next relative relief is less than 0.25.
                & (rr.shift(-1) < 0.25)
                # Moved this to 25 for november, sept and july are at 40
                & (x > 40))
    
    # The toe x coordinate is identified at the first position that satisfies
    # the filter. 
    x_coord = filtered.idxmax()
    # If no positions satistfied the filter, return no toe.
    if filtered[x_coord] == False:
        return None
    else:
        return x_coord


def identify_toe_ip(profile_df, shore_x, crest_x):
    """Returns the dune toe coordinates for the given profile."""    
    # Create a subset of the profile's data to restrict the maximum 
    # change to the desired inflection point
    
    #subset = profile_df[(profile_df.x > 50) & (profile_df.x < 57) & (profile_df.y < 3.2)] # all other time snaps
    subset = profile_df[(profile_df.x > 45) & (profile_df.x < 63) & (profile_df.y < 3.2)] # november
    # Distance values
    x = subset["x"]
    # Elevation values
    y = subset["y"]

    slope = np.degrees((np.arctan(y.shift(1) - y) / (x.shift(1) - x)))
    change = (slope.shift(1)-slope)

    # Create a Truth series for filtering the subset by certain conditions. Add
    # criteria such as "x < 5 crest_x" as needed.
    # IF ADDITIONAL CONDITIONS NEED TO BE ADDED LET BEN KNOW
    #filtered = ((x == x))

    # The toe x coordinate is identified at the position with the largest
    # change in slope.
    x_coord = change.idxmax()
    # If no positions satistfied the filter, return no crest.
    if x_coord == shore_x or x_coord == crest_x:
        return None
    else:
        return x_coord


def identify_toe_poly(profile_xy, shore_x, crest_x):
    """Returns the x coordinate of the dune toe."""
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
    
    
def identify_toe_lcp(profile_df, shore_x, crest_x):
    """Returns the dune toe coordinates for the given profile."""
    # Create a subset of the profile's data from the identified shore
    # to the identified crest. The dune toe can only be found within
    # this subset.
    subset = profile_df.loc[shore_x:crest_x]
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
   

############################### CREST FUNCTIONS ###############################


def identify_crest_rr(profile_df, shore_x):
    """Returns the dune crest coordinates for the given profile."""
    subset = profile_df.loc[shore_x:]
    # Distance values
    x = subset["x"]
    # Elevation values
    y = subset["y"]
    # Relative relief vales
    rr = subset["rr"]

    # Create a Truth Series for filtering the subset by certain conditions.
                #
    filtered = ((x > 30)
                #
                & (x < 85)
                #
                & (rr > 0.55)
                #
                & (y > y.rolling(15).max().shift(-15))
                #
                #& (y > y.rolling(1).max().shift(-1))
                #
                & (rr > rr.rolling(2).max().shift(1))
                #
                & (rr > rr.rolling(2).max.shift(-2)))
        
    # The crest x coordinate is identified at the first position that satisfies
    # the filter.
    x_coord = filtered.idxmax()
    # If no positions satistfied the filter, return no crest.
    if filtered[x_coord] == False:
        return None
    else:
        return x_coord


def identify_crest_standard(profile_df, shore_x):
    """Returns the dune crest coordinates for the given profile."""
    # Create a subset of the profile's data from the identified shore
    # onward. The dune crest can only be found within this subset.
    subset = profile_df.loc[shore_x:]
    # Distance values
    x = subset["x"]
    # Elevation values
    y = subset["y"]

    # Create a Boolean mask for filtering the subset by certain conditions.
            # Current elevation is the largest so far
    mask = ((y > y.shift(1).expanding(min_periods=1).max()) 
            # There is an elevation decrease of more than 2 in the next 20 values
            & (y - y.rolling(20).min().shift(-20) > 2) 
            # Current elevation is greater than the next 10 values.
            & (y > y.rolling(10).max().shift(-10))) 

    # The crest x coordinate is identified at the first position that satisfies
    # the filter. 
    try:
        x_index = mask.idxmax()
    except ValueError:
        return None
    # If no positions satistfied the filter, return None.
    if mask.at[x_index] == False:
        return None
    # Otherwise return the x coordinate.
    else:
        return x.at[x_index]
    


################################ HEEL FUNCTIONS ###############################


def identify_heel_rr(profile_df, crest_x):
    """Returns the dune heel coordinates for the given profile."""
    # Create a subset of the profile's data from the identified crest
    # onward. The dune heel can only be found within this subset.
    subset = profile_df.loc[crest_x:]
    # Distance values
    x = subset["x"]
    # Elevation values
    y = subset["y"]
    # Relative relief values
    rr = subset["rr"]

    # Create a new DataFrame by filtering the data by some conditions. Note
    # that the "~" symbol inverts the filter i.e. the conditions determine what
    # data to exclude, rather than what data to include as seen previously.
                # Horizontal position is 5m past the crest position
    filtered =  ((x - crest_x > 5)
                # Previous 2 relative relief values greater than 0.4.
                & (rr.rolling(2).min().shift(1) > 0.4)
                # Next 2 relative relief values less than 0.4.
                & (rr.rolling(2).max().shift(-2) < 0.4))
                # Add x and y filter criteria as needed
            
    # If no positions satistfied the filter, return None.
    if not filtered.any():
        return None
    else:
        # Returns the x coordinate of the heel. It is identified at the 
        # position with the minimum y value after filtering the y values.
        return y[filtered].idxmin()


def identify_heel_standard(profile_df, crest_x):
    """Returns the dune heel coordinates for the given profile."""
    # Create a subset of the profile's data from the identified crest
    # onward. The dune heel can only be found within this subset.
    subset = profile_df.loc[crest_x:]
    # Distance values
    #x = subset["x"]
    # Elevation values
    y = subset["y"]

    # Create a new DataFrame by filtering the data by some conditions. Note
    # that the "~" symbol inverts the filter i.e. the conditions determine what
    # data to exclude, rather than what data to include as seen previously.
                # There is an elevation change of more than 0.6 in the next 10 values.
    filtered = ~((y - y.rolling(10).min().shift(-10) > 0.6)
                # Current elevation is greater than previous 10.
                & (y > y.rolling(10).max().shift(1))
                # Current elevation is greater than next 10.
                & (y > y.rolling(10).max().shift(-10)))
    # If no positions satistfied the filter, return None.
    if not filtered.any():
        return None
    else:
        # Returns the x coordinate of the heel. It is identified at the 
        # position with the minimum y value after filtering the y values.
        return y[filtered].idxmin()


############################# PROFILE EXTRACTION ##############################


MODES = {"rr" : [identify_shore_standard, identify_toe_rr, identify_crest_rr, identify_heel_rr],
         "rrfar" : [identify_shore_standard, identify_toe_rrfar, identify_crest_rr, identify_heel_rr],
         "ip" : [identify_shore_standard, identify_toe_ip, identify_crest_standard, identify_heel_standard],
         "poly" : [identify_shore_standard, identify_toe_poly, identify_crest_standard, identify_heel_standard],
         "lcp" : [identify_shore_standard, identify_toe_lcp, identify_crest_standard, identify_heel_standard]}


def identify_features(mode): #, shorex=None, toex=None, crestx=None, heelx=None):
    '''
    if shorex is None:
        shore_func = mode[0]
    else:
        shorex = iter(shorex)
        shore_func = lambda a: next(shorex)
    if toex is None:
        toe_func = mode[1]
    else:
        toex = iter(toex)
        toe_func = lambda a, b, c: next(toex)
    if crestx is None:
        crest_func = mode[2]
    else:
        crestx = iter(crestx)
        crest_func = lambda a, b: next(crestx)
    if heelx is None:
        heel_func = mode[3]
    else:
        heelx = iter(heelx)
        heel_func = lambda a: next(heelx)'''

    shore_func, toe_func, crest_func, heel_func = mode


    def _identify_features(profile_xy):
        """
        Returns the coordinates of the shoreline, dune toe, dune crest, and
        dune heel for a given profile as:
        (shore_x, shore_y, toe_x, toe_y, toe_rr, crest_x, crest_y, heel_x, heel_y)
        """
        # Make sure the DataFrame uses the x values as the index. This makes it
        # easy to look up y values corresponding to a given x.
        profile_xy = profile_xy.set_index("x", drop=False)

        # Identify the shore x coordinate if it exists.
        shore_x = shore_func(profile_xy)
        if shore_x is None:
            return (None, ) * 8
        # Identify the crest x coordinate if it exists.
        crest_x = crest_func(profile_xy, shore_x)
        if crest_x is None:
            return (None, ) * 8
        # Identify the toe x coordinate if it exists.
        toe_x = toe_func(profile_xy, shore_x, crest_x)
        if toe_x is None:
            return (None, ) * 8
        # Identify the heel x coordinate if it exists.
        heel_x = heel_func(profile_xy, crest_x)
        if heel_x is None:
            return (None, ) * 8

        # Retrieve the y values for the above features.
        shore_y, toe_y, crest_y, heel_y = profile_xy.loc[[shore_x, toe_x, crest_x, heel_x], "y"]

        return (shore_x, shore_y, toe_x, toe_y, crest_x, crest_y, heel_x, heel_y)

    return _identify_features
