from src.model_trainer import ModelTrainer
from src.data.transformations import sequence_train_test_split

import pandas as pd
import numpy as np
from numpy.typing import NDArray
import os
import re
from timeshap.utils import calc_avg_event
from timeshap.explainer import local_event, local_feat, local_cell_level


class ModelExplainer(ModelTrainer):
    """
    This class extends ModelTrainer to explain the loaded model easier on different datasets.

    It loads model and dataset to initialize the explainer and calculate SHAP values for different cases.

    :param config: configuration for ModelTrainer
    :type config: dict

    "param model: name of model to load
    :type model: str
    """

    def __init__(self, config: dict, model: str, explainer_config: dict):
        super().__init__(model, config)
        self.model_name = model
        dataset_path = os.path.join("data/processed", self.config["dataset"])
        dataset = pd.read_csv(dataset_path)
        self.dataset = dataset
        self.feature_cols = dataset.drop(
            columns=["Default Flag", "time_series", "id"]
        ).columns.to_list()
        plot_feats = dict(zip(self.feature_cols, self.feature_cols))

        self.predictor = self.make_predictor(take_last=1)
        self.baseline_event = self.get_baseline_event()

        self.event_dict = {
            "rs": explainer_config["rs"],
            "nsamples": explainer_config["nsamples"],
        }

        self.feature_dict = {
            "rs": explainer_config["rs"],
            "nsamples": explainer_config["nsamples"],
            "feature_names": self.feature_cols,
            "plot_features": plot_feats,
            "top_feats": explainer_config["top_feats"],
        }

        self.cell_dict = {
            "rs": explainer_config["rs"],
            "nsamples": explainer_config["nsamples"],
            "top_x_events": explainer_config["top_x_events"],
            "top_x_feats": explainer_config["top_x_feats"],
        }

    def get_baseline_event(self) -> pd.DataFrame:
        """Calculate and return baseline event of explainer"""
        X_norm_df = pd.DataFrame(
            self.X_train.reshape(-1, self.X_train.shape[-1]), columns=self.feature_cols
        )
        X_norm_df = X_norm_df[
            X_norm_df["Business Relation Client"] != self.config["padding_value"]
        ].reset_index(drop=True)

        average_event = calc_avg_event(
            X_norm_df, numerical_feats=self.feature_cols, categorical_feats=[]
        )
        return average_event

    def remove_padding(self, X: NDArray) -> NDArray:
        """Calculate original length of an instance and return unpadded sequence"""
        orig_seq_len = X != self.config["padding_value"]
        orig_seq_len = np.sum(np.all(orig_seq_len, axis=2), axis=1)
        X_orig = X[0:1, : orig_seq_len[0], :]
        return X_orig

    def get_event_data(self, X: NDArray) -> pd.DataFrame:
        """Calculate and return event-level SHAP values"""
        event_data = local_event(
            self.predictor,
            self.remove_padding(X),
            self.event_dict,
            None,
            None,
            self.baseline_event,
            0,
        )
        return event_data

    def get_feature_data(self, X: NDArray) -> pd.DataFrame:
        """Calculate and return feature-level SHAP values"""
        feature_data = local_feat(
            self.predictor,
            self.remove_padding(X),
            self.feature_dict,
            None,
            None,
            self.baseline_event,
            0,
        )
        return feature_data

    def get_cell_data(
        self, X: NDArray, event_data: pd.DataFrame, feature_data: pd.DataFrame
    ):
        """Calculate and return cell-level SHAP values"""
        cell_data = local_cell_level(
            self.predictor,
            self.remove_padding(X),
            self.cell_dict,
            event_data,
            feature_data,
            None,
            None,
            self.baseline_event,
            0,
        )
        return cell_data

    def generate_shap_df(self, filename: str):
        y = self.dataset[["id", "Default Flag"]]
        X = self.dataset.drop(columns=["Default Flag", "time_series"])
        X_train, X_test, y_train, y_test = sequence_train_test_split(
            X, y, id_col="id", test_size=self.config["test_size"], random_seed=42
        )
        ids = y_train["id"].unique()
        event_df = pd.DataFrame()
        feature_df = pd.DataFrame()
        for i in range(self.X_train.shape[0]):
            id = ids[i]
            x = self.remove_padding(self.X_train[i : i + 1, :, :])
            pred = self.predictor(x)[0][0]
            event_data = self.get_event_data(x)
            feature_data = self.get_feature_data(x)
            event_data = event_data.rename(columns={"Random seed": "Random Seed"})
            event_data["id"] = id
            event_data["prediction"] = pred
            event_data["Tolerance"] = 0
            event_data["t (event index)"] = event_data["Feature"].apply(
                lambda x: 1 if x == "Pruned Events" else -int(re.findall(r"\d+", x)[0])
            )
            event_df = pd.concat([event_df, event_data], ignore_index=True)
            feature_data = feature_data.rename(columns={"Random seed": "Random Seed"})
            feature_data["id"] = id
            feature_data["prediction"] = pred
            feature_data["Tolerance"] = 0
            feature_df = pd.concat([feature_df, feature_data], ignore_index=True)
        event_filepath = os.path.join(
            "data/model_logs/event_data", f"{filename}_event_data.csv"
        )
        feature_filepath = os.path.join(
            "data/model_logs/feature_data", f"{filename}_feature_data.csv"
        )
        event_df.to_csv(event_filepath, index=False)
        print(f'Event data saved to "{event_filepath}"')
        feature_df.to_csv(feature_filepath, index=False)
        print(f'Feature data saved to "{feature_filepath}"')
