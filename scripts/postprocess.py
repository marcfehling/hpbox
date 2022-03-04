#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
postprocess.py
------------------------
A collection of tools to reduce and summarize data from log files 
generated within the hp-benchmark framework.
"""



import os

import pandas as pd





def build_dataframe(root, extension = "log"):
  '''
  Generate a pandas.DataFrame from a set of log files.
  
  Each block containing data corresponding to a cycle will make up
  one row in the DataFrame object.
  
  All files with the file extenstion 'extension' in the folder 'root'
  will be considered. Nested directories will be ignored.

  Parameters
  ----------
  root : string
    Path to the directory that will be scanned for files. No nested
    directories will be considered.
  extension : string
    Only files with this extension will be considered.

  Returns
  -------
  df : pandas.DataFrame
    DataFrame containing data from all considered log files.
  '''
  # get list of all files
  filenames = [os.path.join(root, f) for f in os.listdir(root)
               if f.lower().endswith('.'+extension)]
    
  # read first dataframe entirely and use first row as header
  df = [pd.read_table(f, delim_whitespace=True) for f in filenames]
  return pd.concat(df, ignore_index=True)



def reduce_dataframe(df, multiindex, parameters, check_errors=True):
  '''
  Customize a large pandas.DataFrame into a smaller one to be able to
  work on just a fraction of the complete dataset.
  
  The following operations will be performed on a copy of the provided
  pandas.DataFrame which will be returned by this function:
   - Keep rows that correspond to values of 'parameters'.
   - Drop columns that correspond to keys of 'parameters'.
   - Group DataFrame according to 'multiindex'.
   - If multiple rows exists for the same 'multiindex', the minimum over
     all ambiguous entries will be picked for each data column.
     The index columns specified by 'multiindex' will be left untouched.
  
  If there are ambiguous rows to the same multiindex, we take their
  minimum for each column since most entries correspond to walltimes and
  we want to exclude the influence of any OS overhead.
  
  However, the data corresponding to errors should be identical for all
  ambiguous rows since they match the same setup. We specifically check
  whether entries for rows 'L2 error' and 'H1 error' are unique. This
  should be disabled when multiple processors were used by setting the
  parameter check_errors accordingly.
  
  Parameters
  ----------
  df : pandas.DataFrame
    DataFrame to work with. Will not be changed.
  multiindices : list
    All columns that will be treated as indices for the returned
    DataFrame.
  parameters : dict
    Dictionary specifying certain parameters. Only those entries
    corresponding to these parameters will be considered.
    Keys correspond to columns, while values conform to matching rows.
    Keys are not allowed to also be part of multiindices.
  check_errors : bool
    Allow checking whether entries for rows 'L2 error' and 'H1 error'
    are unique.
  
  Returns
  -------
  df_reduced : pandas.DataFrame
    DataFrame that has been reduced according to the documentation.
  '''
  # select certain rows:
  # filter dataframe using parameters dict.
  #   see: https://stackoverflow.com/questions/34157811/filter-a-pandas-dataframe-using-values-from-a-dict
  df_reduced = df.loc[(df[list(parameters)] == pd.Series(parameters)).all(axis=1)]
  
  # reduce number of columns:
  # drop these columns including the time one
  df_reduced.drop(list(parameters), axis='columns', inplace=True)
  
  # assign multiindices
  df_reduced.set_index(multiindex, inplace=True)
  df_reduced = df_reduced.groupby(multiindex)
  
  # up to this point, all entries corresponding to the same multiindex
  # should have produced the same data, only walltimes should vary.
  
  # verify that these ambiguous entries correspond to the same error
  # by checking whether they are unique in the dataframe.
  #   NOTE: I expect slight differences in using multiple processors, 
  #         so we may skip these assertion or check within a range...
  if check_errors:
    assert(df_reduced['L2 error'].nunique().eq(1).all())
    assert(df_reduced['H1 error'].nunique().eq(1).all())
  
  # minimum over all entries corresponding to the same multiindex
  return df_reduced.min()



def plot_dataframe(df, xylist):
  '''
  Present columns from a pandas.DataFrame in a double logarithmic plot.

  Parameters
    ----------
    df : pandas.DataFrame
      DataFrame to work with. Will not be changed.
    xylist : list of tuples of length 2 of strings (sx, sy)
      List of datasets to be displayed. Each column sy will be plotted
      against the corresponding column sx.
  '''
  df_plot = df.reset_index().astype(float)
  
  ax = plt.axes()
  for xy in xylist:
    df_plot.plot(x=xy[0], y=xy[1], loglog=True, ax=ax)





if __name__ == '__main__':
  import sys
  import matplotlib.pyplot as plt
  
  '''
  BUILD DATAFRAME
  '''
  # generate dataframe from all logfiles in specified folder
  root = "../build/"
  extension = "log"
  argc = len(sys.argv)
  if argc > 1:
    root = sys.argv[1]
  if argc > 2:
    extension = sys.argv[2]
  df_full = build_dataframe(root, extension)
  
  # write out full dataframe (if desired)
  # df_full.to_csv("dataframe.csv")
  # df_full.to_pickle("dataframe.pkl")
  
  
  '''
  REDUCE DATAFRAME
  '''
  # read in previously gathered dataframe (if desired)
  # df_full.read_csv("dataframe.csv")
  # df_full.read_pickle("dataframe.pkl")
  
  # reduce dataframe 
  df_reduced = reduce_dataframe(
    df_full,
    multiindex = ["ncpus"],
    parameters = {"stem":"stokes_y-pipe-scaling"},
    check_errors = False)
  
  # write out reduced dataframe
  df_reduced.to_csv('scaling.csv')
  
  
  '''
  PLOT RESULTS
  '''
  plot_dataframe(df_reduced, [('ncpus', 'solve'), ('ncpus', 'assemble_system')])
  plt.show()
