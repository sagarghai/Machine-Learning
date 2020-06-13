import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from pandas.api.types import is_string_dtype, is_numeric_dtype
import seaborn as sns


def display_all(df):
    """
    This function takes a pandas Dataframe and prints all the values in it
    upto a max of 1000 rows and columns

    Input:
    ------
    df: Pandas dataframe
    """
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


def handle_dates(df, c, time=False, drop=True):
    """
    This function takes a dataframe and splits the date variable into its
    properties that are consistent with the pandas datetime object. Each property
    can be accessed by the <column_name>_<attribute> attribute.

    Input:
    -----
    df: Pandas dataframe object
    c: datetype column name
    time: whether the datetime object contains time attributes
    drop: Whether to drop the datetime column after splitting into attributes
    """
    columns = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
               'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time:
        columns = columns + ['Hour', 'Minute', 'Second']
    for n in columns:
        df[c + "_" + n] = getattr(df[c].dt, n.lower())
    if drop:
        df.drop(c, inplace=True, axis=1)


def handle_categories(df):
    """
    Change string type columnsto category type columns.

    Input:
    -----

    df: Pandas dataframe object
    """
    for n, c in df.items():
        if is_string_dtype(c):
            df[n] = c.astype('category').cat.as_ordered()


def apply_cats(df, trn):
    """
    This function applies the categories found in training set to the validation/testing set.
    We assume that nothing new is discovered in the validation set.

    Input:
    -----

    df: Pandas Dataframe object
    trn: the training set object(pandas df)
    """
    for n, c in df.items():
        if df[n].dtype.name == 'category':
            df[n] = pd.Categorical(c, categories=trn[n].cat.categories, ordered=True)


def handle_missing(df, nas={}):
    """
    This function takes care of the missing values in the dataset. For categorical types, it adds
    a column which denotes whether nan or not, for the others it replaces them with medians.

    Input:
    -----

    df: Pandas Dataframe object
    nas: dictionary of values for the missing columns

    Returns:
    -------
    Dictionary of missing values.
    """
    for n, c in df.items():
        if is_numeric_dtype(c):
            if n not in nas:
                nas[n] = c.median()
            df[n].fillna(nas[n], inplace=True)
        if c.dtype.name == 'category':
            if c.isna().sum() > 0:
                df[n + "_na"] = c.isna()
    return nas


def numericalise(df):
    """
    Changes each category type column into codes and adds 1

    Input:
    -----

    df: Pandas Dataframe Object
    """
    for n, c in df.items():
        if c.dtype.name == 'category':
            df[n] = c.cat.codes + 1


def split_vals(df, n):
    """
    This function splits a given dataframe or an array into two parts at n.

    Input:
    -----

    df: Pandas Dataframe Object
    n: The index at which to split.
    """
    return df[:n], df[n:]
