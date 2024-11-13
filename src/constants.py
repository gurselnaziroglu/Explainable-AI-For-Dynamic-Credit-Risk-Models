"""Shared constants and custom types are stored here."""

import os
from enum import Enum


class DatasetUsed(str, Enum):
    """Used for dataset selection."""

    DEFAULT = "credit_risk_dataset_processed[2006,2022].csv"
    DEFAULT_TILL_2018 = "credit_risk_dataset_processed[2006,2018].csv"
    DEFAULT_POST_2018 = "credit_risk_dataset_processed[2019,2022].csv"

    DEFAULT_2K = "credit_risk_dataset_2k_processed[2006,2022].csv"
    DEFAULT_2K_TILL_2018 = "credit_risk_dataset_2k_processed[2006,2018].csv"
    DEFAULT_2K_POST_2018 = "credit_risk_dataset_2k_processed[2019,2022].csv"

    DEFAULT_5K = "credit_risk_dataset_5k_processed[2006,2022].csv"
    DEFAULT_5K_TILL_2018 = "credit_risk_dataset_5k_processed[2006,2018].csv"
    DEFAULT_5K_POST_2018 = "credit_risk_dataset_5k_processed[2019,2022].csv"


class ImbalanceHandling(str, Enum):
    """Used for imbalance handling method selection."""

    CLASS_WEIGHTS = "class_weights"
    RANDOM_OVER_SAMPLING = "random_over_sampling"


class ProjectRoot(str, Enum):
    """Used for reaching project root path."""

    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DeepLearningModelType(str, Enum):
    """Used for differentiating the current driver model."""

    BI_LSTM = "BI_LSTM"
    DIFFERENT_SEQUENCES_LSTM = "DIFFERENT_SEQUENCES_LSTM"
