"""This module contains the helper functions used in imputing missing data."""
import glob
import os

import numpy as np
import pandas as pd
import pycountry

UNEMPLOYMENT_FILE_ID = "unemployment"


def get_unemployment_rate(unemployment_df, year, country_alpha_2):
    """
    Return the unemployment rate given the year and the country 2-letter abbreviation.

    :param unemployment_df: the dataframe containing the unemployment rate data.
    :type unemployment_df: pandas.DataFrame
    :param year: the year.
    :type year: int
    :param country_alpha_2: the 2-letter abbreviated country name (ex. CH, LT).
    :type country_alpha_2: str
    ...
    :return: the unemployemnt rate for this country in that year.
    :rtype: float
    """
    country_entity = pycountry.countries.get(alpha_2=country_alpha_2)
    if country_entity is None:
        return 0.0
    else:
        country_name = country_entity.name
        value = unemployment_df.loc[
            (unemployment_df["CountryName"] == country_name)
            & (unemployment_df["PeriodCode"] == year),
            "Value",
        ].iloc[0]
        try:
            parsed_value = float(value)
            return parsed_value / 100.0
        except ValueError:
            return 0.0


def fill_unemployment_country(unemployment_df):
    """
    Return a function to replace null values of unemployment country.

    :param unemployment_df: the dataframe containing the unemployment rate data.
    :type unemployment_df: pandas.DataFrame
    ...
    :return: A pandas dataframe with unemployment country trend imputed.
    :rtype: pandas.DataFrame
    """

    def fill_function(row):
        if pd.isnull(row["Unemployment country"]):
            row_year = row.name.year
            row_country = row["CMS Country"]
            return get_unemployment_rate(unemployment_df, row_year, row_country)
        else:
            return row["Unemployment country"]

    return fill_function


def fill_unemployment_country_trend(unemployment_df):
    """
    Return a function to replace null values of unemployment country trend.

    according to the following formula:
    (unemployment_rate - previous_unemployment_rate) / previous_unemployment_rate

    Values are rounded to 3 decimal places.

    if not applicaple fill with 0.0.

    :param unemployment_df: the dataframe containing the unemployment rate data.
    :type unemployment_df: pandas.DataFrame
    ...
    :return: A pandas dataframe with unemployment country trend imputed.
    :rtype: pandas.DataFrame
    """

    def get_relative_unemployment(row):
        relative = row["Unemployment country trend (relative)"]
        if pd.isnull(relative):
            unemployment = row["Unemployment country"]
            row_country = row["CMS Country"]
            prev_year = row.name.year - 1
            unemployment_prev = get_unemployment_rate(
                unemployment_df, prev_year, row_country
            )

            return (
                float(
                    np.round((unemployment - unemployment_prev) / unemployment_prev, 3)
                )
                if unemployment_prev != 0.0
                else 0.0
            )
        else:
            return relative

    return get_relative_unemployment


def fill_object_value_change(df):
    """
    Fill null values of object value change.

    according to the following formula:
    row["Object Value Change"] = row_3_years_later["Object Value Change"] - row_3_years_later["Object Value Change 3 Year"]

    if not applicaple fill with mean if the group which this row belongs to.

    :param df: the dataframe used for the data.
    :type df: pandas.DataFrame
    ...
    :return: A pandas dataframe with object value change imputed.
    :rtype: pandas.DataFrame
    """
    for id in df["id"].unique():
        sequence = df.loc[df["id"] == id].sort_index()
        for date, row in sequence.iterrows():
            row_year = date.year
            row_plus_3 = sequence.loc[sequence.index.year == row_year + 3]

            if pd.isnull(row["Object Value Change"]):
                if len(row_plus_3) > 0 and row_year == 2006:
                    df.at[date, "Object Value Change"] = (
                        row_plus_3["Object Value Change"]
                        - row_plus_3["Object Value Change 3 Year"]
                    )
                else:
                    df.at[date, "Object Value Change"] = sequence[
                        "Object Value Change"
                    ].mean()
    return df


def handle_missing_values(df, external_source):
    """
    Handle the known missing value problems in the dataset.

    :param df: the dataframe used for the data.
    :type df: pandas.DataFrame

    :param external_source: the path for any external data needed.
    :type external_source: str
    ...
    :return: A new pandas dataframe with missing values imputed
    :rtype: pandas.DataFrame
    """
    unemployment_df = None
    for dataset_path in glob.glob(os.path.join(external_source, "*.csv")):
        filename = os.path.basename(dataset_path)
        if UNEMPLOYMENT_FILE_ID in filename:
            unemployment_df = pd.read_csv(dataset_path)[
                ["CountryName", "PeriodCode", "Value"]
            ]

    assert unemployment_df is not None, "could not find unemployment rate data"
    # Unemployment country mssing values are replaced with data from Unicef

    df_imputed = df.copy(deep=True)
    df_imputed["Unemployment country"] = df_imputed.apply(
        fill_unemployment_country(unemployment_df), axis=1
    )

    # Unemployment country mssing values are replaced with data according to the formula in the data description
    df_imputed["Unemployment country trend (relative)"] = df_imputed.apply(
        fill_unemployment_country_trend(
            unemployment_df,
        ),
        axis=1,
    )

    # Object Value Change 3 Year is imputed with future values. If they do not exist then mean of group
    df_imputed = fill_object_value_change(df_imputed)
    df_imputed["Object Value Change 3 Year"] = df_imputed[
        "Object Value Change 3 Year"
    ].fillna(df_imputed["Object Value Change"])

    # impute with mean. Property Type works as indicator variable of missing rows.
    df_imputed["Tenant PD"] = df_imputed["Tenant PD"].replace({-9999.0: np.nan})
    df_imputed["Tenant PD"] = df_imputed["Tenant PD"].fillna(
        df_imputed["Tenant PD"].mean()
    )

    df_imputed["Vacancy rate"] = df_imputed["Vacancy rate"].replace({-9999.0: np.nan})
    df_imputed["Vacancy rate"] = df_imputed["Vacancy rate"].fillna(
        df_imputed["Vacancy rate"].mean()
    )

    return df_imputed
