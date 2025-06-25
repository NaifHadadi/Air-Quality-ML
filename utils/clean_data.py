# Importing the Imp libraries for model making and analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy


def lower_and_strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    """lowercase and strip whitespace from column names"""
    df.columns = df.columns.str.lower().str.strip()
    return df


def strip_whitespace_from_name(df: pd.DataFrame) -> pd.DataFrame:
    """strip whitespace from each entry in the 'name' column"""
    if "name" in df.columns:
        df["name"] = df["name"].str.strip()
        # Cleans data values  in the 'name' column
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """clean the dataframe by dropping unnecessary columns, removing nulls and duplicates"""
    df = df.drop(["site", "variable", "units", "quality", "prelim"], axis=1)
    # axis =1 mean columns. columns removed for redundancy, site count already validated against name count
    df = df.dropna()  # remove empty rows
    df = df.drop_duplicates()  # remove duplicate rows
    return df


def remove_negatives(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
    """Removes negative values from specified columns in air quality DataFrames.
    If no columns are specified, it removes negatives from all numeric columns.

    Input: remove_negatives(df,columns=['Name']), if not specify them all columns
    """

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    mask = (df[columns] >= 0).all(axis=1)

    # Calculate the number of rows removed
    removed_rows = len(df) - mask.sum()

    # Display the number of rows removed
    print(f"The number of removed rows is: {removed_rows}")

    return df[mask]


def get_season(month_col):
    if month_col in [3, 4, 5]:
        return "Spring"
    elif month_col in [6, 7, 8]:
        return "Summer"
    elif month_col in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"


def add_date_columns(
    df: pd.DataFrame, date_column: str = None, columns: list[str] = None
):
    # Add year, month, and season columns
    if date_column == "Index":
        df["year"] = df.index.year
        df["month"] = df.index.month
        df["season"] = df.index.month.map(get_season)
    else:
        if "year" not in df.columns and "year" in columns:
            df["year"] = df[date_column].dt.year

        if "month" not in df.columns and "year" in columns:
            df["month"] = df[date_column].dt.month

        if "season" not in df.columns and "year" in columns:
            df["season"] = df["month"].apply(get_season)
    return df


def aggregate_to_daily_mean(
    df: pd.DataFrame,
    date_column: str = None,
    aggregate_by: list[str] = None,
    columns_to_aggregate: list[str] = None,
    start_year: str = None,
    end_year: str = None,
) -> pd.DataFrame:
    """Aggregates the DataFrame by calculating daily mean, adding year, month, season,
    and limits the timeframe to a specified range."""

    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])
    # Filter the DataFrame by the specified time range
    df_filtered = df[(df[date_column] >= start_year) & (df[date_column] <= end_year)]

    # Group by the provided columns and calculate the mean
    df_grouped = (
        df_filtered.groupby(aggregate_by)[columns_to_aggregate]
        .agg("mean")
        .reset_index()
    )

    # Sort by date column
    df_grouped = df_grouped.sort_values(by=date_column)

    return df_grouped


def remove_outliers(df, columns, threshold=4):
    """
    Remove outliers from specified columns in a DataFrame using Z-score and print the number of removed rows for each column.

    Parameters:
    - df: pandas DataFrame
    - columns: List of column names to check for outliers
    - threshold: Z-score threshold to define outliers (default is 4)

    Returns:
    - DataFrame with outliers removed
    """
    original_shape = (
        df.shape
    )  # Store original shape to calculate number of removed rows

    for column in columns:
        if column in df.columns:
            z_scores = np.abs(scipy.stats.zscore(df[column]))
            # Identify rows to be removed
            rows_before = df.shape[0]
            df = df[z_scores <= threshold]
            rows_removed = rows_before - df.shape[0]

            # Print the number of removed rows for this column
            print(f"Removed {rows_removed} rows due to outliers in column '{column}'")
        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")

    return df


def apply_data_cleaning(df: pd.DataFrame, function_name: str, **kwargs) -> pd.DataFrame:
    cleaning_functions = {
        "lower_and_strip_cols": lower_and_strip_cols,
        "strip_whitespace_from_name": strip_whitespace_from_name,
        "clean_data": clean_data,
        "remove_negatives": remove_negatives,
        "aggregate_to_daily_mean": aggregate_to_daily_mean,
        "add_date_columns": add_date_columns,
        "remove_outliers": remove_outliers,
    }

    func = cleaning_functions.get(function_name)

    return func(df, **kwargs)  # Pass kwargs to the function
