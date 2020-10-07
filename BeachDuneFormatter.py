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

