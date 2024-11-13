"""Functions used for data loading."""

import os

import pandas as pd
from numpy.typing import NDArray
from src.constants import ImbalanceHandling


from src.data.transformations import (
    get_sequences,
    sequence_train_test_split,
    random_oversampling,
)


def load_default_dataset(
    file_name: str,
    val_size: float = 0.2,
    test_size: float = 0.3,
    *,
    scaler=None,
    scale_excluded_columns: set[str] = set(),
    imbalance_handling: ImbalanceHandling | None = None,
    fit_scaler: bool = True,
) -> tuple[
    list[list[list[float]]],
    list[list[list[float]]],
    list[list[list[float]]],
    list[list[list[float]]],
]:
    """
    Load the default dataset and prepare its train and test sequences.

    the following is done in addition to the loading:
    1- Train/Test split
    2- Over/Undersample data (optional)
    3- Standard Scaling (if a scaler is provided)
    4- Transform to sequences

    :param file_name: the name of the dataset file inside the processed directory.
    :type file_name: str

    :param val_size: the portion of the dataset reserved for validation [0.0, 1.0].
    :type val_size: float

    :param test_size: the portion of the dataset reserved for testing [0.0, 1.0].
    :type test_size: float

    :param id_col: name of the column containing the ids.
    :type id_col: str

    :param scaler: instance of the scaler that will be used. Should have fit_transform() and transform() functions.

    :param scale_excluded_columns: the set of the columns excluding from scaling.
    :type scale_excluded_columns: set[str]

    :param imbalance_handling: class imbalance handling method.
    :type imbalance_handling: ImbalanceHandling | None

    :param fit_scaler: fit the given scaler.
    :type fit_scaler: bool
    ...
    :return: X_train, X_val, X_test, y_train, y_val, y_test sequence lists
    :rtype: tuple[list[list[list[float]]], list[list[list[float]]], list[list[list[float]]], list[list[list[float]]], list[list[list[float]]], list[list[list[float]]]]
    """
    dataset_path = os.path.join("data/processed", file_name)

    dataset = pd.read_csv(dataset_path)
    dataset["time_series"] = pd.to_datetime(dataset["time_series"])
    dataset = dataset.sort_values(by=["id", "time_series"])

    # traget separation
    y = dataset[["id", "Default Flag"]]
    X = dataset.drop(columns=["Default Flag", "time_series"])

    X_train, X_test, y_train, y_test = sequence_train_test_split(
        X, y, id_col="id", test_size=test_size, random_seed=42
    )

    X_train, X_val, y_train, y_val = sequence_train_test_split(
        X_train, y_train, id_col="id", test_size=val_size, random_seed=42
    )

    if imbalance_handling == ImbalanceHandling.RANDOM_OVER_SAMPLING:
        X_train, y_train = random_oversampling(X_train, y_train)
    else:
        ValueError(f"Imbalance handling using {imbalance_handling} is not supported")

    if scaler is not None:
        scaled_columns = list(set(X_train.columns) - scale_excluded_columns)
        if len(X_train) > 0:
            X_train.loc[:, scaled_columns] = (
                scaler.fit_transform(X_train[scaled_columns])
                if fit_scaler
                else scaler.transform(X_train[scaled_columns])
            )

        if len(X_val) > 0:
            X_val.loc[:, scaled_columns] = scaler.transform(X_val[scaled_columns])

        if len(X_test) > 0:
            X_test.loc[:, scaled_columns] = scaler.transform(X_test[scaled_columns])

    X_train, X_val, X_test, y_train, y_val, y_test = get_sequences(
        X_train, X_val, X_test, y_train, y_val, y_test, id_col="id"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
