"""This module has functions for sequence transformations."""

from copy import deepcopy

import numpy as np
import pandas as pd


def sequence_train_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    id_col: str = "id",
    test_size: float = 0.3,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform train/test split on dataframes based on customers/ids instead of rows.

    :param X: the dataframe used for the features.
    :type X: pandas.DataFrame

    :param y: the dataframe used for the outputs. Should contain 2 the id column as well.
    :type y: pandas.DataFrame

    :param id_col: name of the column containing the ids.
    :type id_col: str

    :param test_size: the portion of the dataset reserved for testing [0.0, 1.0].
    :type test_size: float

    :param random_seed: the numpy random seed.
    :type random_seed: int
    ...
    :return: X_train, X_test, y_train, y_test dataframes
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    unique_ids = np.array(X[id_col].unique())

    n = len(unique_ids)
    train_n = n - int(n * test_size)

    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_ids)

    train_ids = unique_ids[:train_n]
    test_ids = unique_ids[train_n:]

    X_train = X.loc[X[id_col].isin(train_ids)]
    y_train = y.loc[y[id_col].isin(train_ids)]

    X_test = X.loc[X[id_col].isin(test_ids)]
    y_test = y.loc[y[id_col].isin(test_ids)]

    return X_train, X_test, y_train, y_test


def random_oversampling(
    X: pd.DataFrame,
    y: pd.DataFrame,
    id_col: str = "id",
    label_col: str = "Default Flag",
    minority_class: int = 1,
    random_seed: int = 42,
    data_augmentation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform random oversampling to increase minority class sequence to equal to number of majority class sequences.

    :param X: the dataframe used for the features.
    :type X: pandas.DataFrame

    :param y: the dataframe used for the outputs. Should contain 2 the id column as well.
    :type y: pandas.DataFrame

    :param id_col: name of the column containing the ids.
    :type id_col: str

    :param label_col: name of the column containing the class label.
    :type label_col: str

    :param minority_class: value of the minority label
    :type minority_class: int

    :param random_seed: pandas.sample random seed.
    :type random_seed: int

    :param data_augmentation: Flag to denote whether to perform augmentation oversampled sequences by randomly trim lengths
    :type data_augmentation: bool
    ...
    :return: X_resampled, y_resampled
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    max_id = X[id_col].max()
    n_ids = X[id_col].nunique()
    id_flags = y.groupby([id_col])[label_col].max()
    n_minority = id_flags.value_counts()[minority_class]
    n_majority = n_ids - n_minority
    to_sample = n_majority - n_minority

    minority_sampled_ids = y[y[label_col] == minority_class].sample(
        n=to_sample, replace=True, random_state=random_seed
    )[["id"]]
    minority_sampled_ids["new_id"] = (
        minority_sampled_ids.reset_index().index + max_id + 1
    )
    X_sampled = minority_sampled_ids.merge(X, on=[id_col], how="left")
    y_sampled = minority_sampled_ids.merge(y, on=[id_col], how="left")

    if data_augmentation:
        X_sampled["seq_length"] = (
            X_sampled.groupby(["new_id"]).cumcount(ascending=False) + 1
        )
        y_sampled["seq_length"] = (
            y_sampled.groupby(["new_id"]).cumcount(ascending=False) + 1
        )

        X_sampled_length = (
            X_sampled.groupby(["new_id"])[id_col]
            .count()
            .reset_index()
            .rename(columns={id_col: "max_length"})
        )
        X_sampled_length["new_length"] = X_sampled_length["max_length"].apply(
            lambda x: np.random.randint(1, x + 1)
        )

        X_sampled = X_sampled.merge(X_sampled_length[["new_id", "new_length"]])
        X_sampled = X_sampled[X_sampled["seq_length"] <= X_sampled["new_length"]].drop(
            ["seq_length", "new_length"], axis=1
        )
        y_sampled = y_sampled.merge(X_sampled_length[["new_id", "new_length"]])
        y_sampled = y_sampled[y_sampled["seq_length"] <= y_sampled["new_length"]].drop(
            ["seq_length", "new_length"], axis=1
        )

    X_sampled[id_col] = X_sampled["new_id"]
    X_sampled = X_sampled.drop("new_id", axis=1)

    y_sampled[id_col] = y_sampled["new_id"]
    y_sampled = y_sampled.drop("new_id", axis=1)

    X_resampled = pd.concat([X, X_sampled])
    y_resampled = pd.concat([y, y_sampled])
    return X_resampled, y_resampled


def get_sequences(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    y_test: pd.DataFrame,
    *,
    id_col: str = "id",
) -> tuple[
    list[list[list[float]]],
    list[list[list[float]]],
    list[list[list[float]]],
    list[list[list[float]]],
]:
    """
    Convert dataframes to sequence lists.

    :param X_train: the dataframe of the training features.
    :type X_train: pandas.DataFrame

    :param X_val: the dataframe of the validation features.
    :type X_val: pandas.DataFrame

    :param X_test: the dataframe of the testing features.
    :type X_test: pandas.DataFrame

    :param y_train: the dataframe of the training outputs, should contain 2 columns id and response.
    :type y_train: pandas.DataFrame

    :param y_val: the dataframe of the validation outputs, should contain 2 columns id and response.
    :type y_val: pandas.DataFrame

    :param y_test: the dataframe of the testing outputs, should contain 2 columns id and response.
    :type y_test: pandas.DataFrame

    :param id_col: name of the column containing the ids.
    :type id_col: str
    ...
    :return: X_train, X_val, X_test, y_train, y_val, y_test sequence lists
    :rtype: tuple[list[list[list[float]]], list[list[list[float]]], list[list[list[float]]], list[list[list[float]]], list[list[list[float]]], list[list[list[float]]]]
    """
    X_train_sequences = [
        group.values[:, 1:].tolist() for _, group in X_train.groupby("id")
    ]
    X_val_sequences = [group.values[:, 1:].tolist() for _, group in X_val.groupby("id")]
    X_test_sequences = [
        group.values[:, 1:].tolist() for _, group in X_test.groupby("id")
    ]

    y_train_sequences = [
        group.values[:, 1:2].tolist() for _, group in y_train.groupby("id")
    ]
    y_val_sequences = [
        group.values[:, 1:2].tolist() for _, group in y_val.groupby("id")
    ]
    y_test_sequences = [
        group.values[:, 1:2].tolist() for _, group in y_test.groupby("id")
    ]

    return (
        X_train_sequences,
        X_val_sequences,
        X_test_sequences,
        y_train_sequences,
        y_val_sequences,
        y_test_sequences,
    )


def pad_sequences(
    sequences: list[list[list[float]]],
    *,
    max_length: int,
    feature_length: int,
    padding_value: float,
) -> list[list[list[float]]]:
    """
    Pad a sequence list with a specific value.

    :param sequences: the list if sequences to be padded.
    :type sequences: list[list[list[float]]]

    :param max_length: the maximum length of the sequence.
    :type max_length: int

    :param feature_length: the feature length for each element in a sequence.
    :type feature_length: int

    :param padding_value: the value of the padding to be added.
    :type padding_value: int
    ...
    :return: the padded sequence list
    :rtype: list[list[list[float]]]
    """
    sequences_out = deepcopy(sequences)
    for i in range(len(sequences_out)):
        max_positive_length = max(max_length or len(sequences_out[i]), 0)
        padding_length = max(
            (max_length or len(sequences_out[i])) - len(sequences_out[i]), 0
        )

        sequences_out[i] = (
            sequences_out[i]
            + [
                [padding_value for _ in range(feature_length)]
                for _ in range(padding_length)
            ]
        )[-max_positive_length:]

    return sequences_out
