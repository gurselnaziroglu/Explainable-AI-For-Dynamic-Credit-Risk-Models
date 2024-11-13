"""This module contains the helper functions used in ."""
import glob
import os

import numpy as np
import pandas as pd

COUNTRIES_FILE_ID = "countries"


def encode_categotical_values(df, external_source):
    """
    Encode categorical values.

    :param df: the dataframe used for the data.
    :type df: pandas.DataFrame

    :param external_source: the path for any external data needed.
    :type external_source: str
    ...
    :return: A new pandas dataframe with categprical values encoded
    :rtype: pandas.DataFrame
    """
    countries_df = None
    for dataset_path in glob.glob(os.path.join(external_source, "*.csv")):
        filename = os.path.basename(dataset_path)
        if COUNTRIES_FILE_ID in filename:
            countries_df = pd.read_csv(dataset_path, index_col="country")

    assert countries_df is not None, "could not find country coordinate data"

    # property type is one-hot encoded to be 1 for commercial and 0 for residential properties
    df["Property Type"] = df["Property Type"].map(
        {"commercial": 1.0, "residential": 0.0}
    )
    df = df.rename(columns={"Property Type": "Commercial Property"})

    # CMS Country is encoded as longitude and latitude coordinates
    df["CMS Country Longitude"] = df["CMS Country"].apply(
        lambda value: float(countries_df.loc[value, ["longitude"]].iloc[0])
    )
    df["CMS Country Latitude"] = df["CMS Country"].apply(
        lambda value: float(countries_df.loc[value, ["latitude"]].iloc[0])
    )
    df = df.drop(columns=["CMS Country"])

    return df
