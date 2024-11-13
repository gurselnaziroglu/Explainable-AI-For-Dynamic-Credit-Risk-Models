"""Contains the logic of tuning the bi_lstm on future data."""

from src.constants import DatasetUsed, DeepLearningModelType, ImbalanceHandling
from sklearn.preprocessing import StandardScaler
from src.helpers import MacroF1
from src.model_trainer import ModelTrainer
from src.models.bidirectional_lstm.bi_lstm_script import get_model
from src.data.process_dataset import process_dataset_call
from src.models.genetic_algorithm.GA import genetic_algorithm
from src.models.bidirectional_lstm.bi_lstm_rl_evaluation import set_evaluation_config
import numpy as np
import re
import os
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import math


class Retrainer:
    """Class that simulates the passage of time. Tunes the model on future data."""

    def __init__(
        self,
        model_type: DeepLearningModelType,
        model_config: dict,
        lookback_window: int,
    ) -> None:
        """
        Construct a new instance.

        :param model_type: the type of the given model, currently only supports bi-lstm
        :type model_type: DeepLearningModelType

        :param model_config: the config for the current model
        :type model_config: dict

        :param lookback_window: when retraining on a new year, give the model a sequence of size lookback_window.
        :type lookback_window: int
        """
        if model_type == DeepLearningModelType.BI_LSTM:
            self.config = model_config
            self.model_trainer = ModelTrainer(get_model(-100), self.config)
        else:
            raise NotImplementedError(
                f"Retrainer class currently does not support the {model_type} model"
            )

        self.lookback_window = lookback_window
        self.model_type = model_type
        self.step = 0
        self.datasets = self._prepare_datasets()

    def _prepare_datasets(
        self,
    ) -> list[
        tuple[
            str,
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
    ]:
        """
        Return all datasets the model will be evaluated on.

        Process the datasets if not processed.

        ...
        :return: list of the datasets, each dataset is a tuple (name, X_train, X_val, X_test, y_train, y_val, y_test)
        """
        dataset_name = self.config["dataset"]

        training_dataset = (
            dataset_name,
            self.model_trainer.X_train,
            self.model_trainer.X_val,
            self.model_trainer.X_test,
            self.model_trainer.y_train,
            self.model_trainer.y_val,
            self.model_trainer.y_test,
        )

        assert (
            "processed" in dataset_name
        ), "The Retrainer should take the name of a processed dataset for the initial training"

        split_name = dataset_name.split("_processed")
        dataset_prefix = split_name[0]
        initial_start, initial_end = re.findall(r"\d+", split_name[1])
        initial_start, initial_end = int(initial_start), int(initial_end)

        retraining_periods = [
            (period_end - self.lookback_window + 1, period_end)
            for period_end in range(initial_end + 1, 2023)
        ]

        processed_dataset_paths = []

        for start, end in retraining_periods:
            current_processed_path = (
                f"data/processed/{dataset_prefix}_processed[{start},{end}].csv"
            )

            if not os.path.isfile(current_processed_path):
                current_processed_path = process_dataset_call(
                    src="data/raw",
                    dest="data/processed",
                    start=start,
                    end=end,
                    dataset_file_name=f"{dataset_prefix}.csv",
                )[0]
            processed_dataset_paths.append(current_processed_path)

        future_data = []
        for path in processed_dataset_paths:
            future_dataset_name = os.path.basename(path)
            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
            ) = self.model_trainer.prepare_dataset(
                future_dataset_name,
                val_size=self.config["validation_size"],
                test_size=self.config["test_size"],
                fit_scaler=False,
                load=False,
            )
            future_data.append(
                (future_dataset_name, X_train, X_val, X_test, y_train, y_val, y_test)
            )

        return [training_dataset, *future_data]

    def train_current_dataset(self) -> None:
        """
        Tune the model on the current dataset using the GA-RL solution.

        Saves the best model.
        """
        period = self.get_current_period()

        # run GA-RL and save the best model
        set_evaluation_config(self.model_trainer, period)
        genetic_algorithm(
            population_size=10,
            generations=3,
            deep_learning_model_type=DeepLearningModelType.BI_LSTM,
        )

        # load the best model to the trainer for future tuning
        self.model_trainer.load(f"final_models/bi_lstm{period}")

    def advance(self) -> None:
        """Advance to the next dataset, simulates passage of time."""
        n = len(self)
        if self.step == n - 1:
            print("No further daasets left")
        else:
            self.step += 1
            self.model_trainer.assign_dataset(*self.datasets[self.step])

    def get_current_period(self) -> list[int]:
        """
        Use the current dataset name to deduce the start and end year for this training period.

        ...
        :return: the start and end year as a list of length 2.
        :rtype: list[int]
        """
        dataset_name = self.datasets[self.step][0]
        start, end = re.findall(r"\d+", dataset_name)[1:]
        return [int(start), int(end)]

    def plot_confusion_for_all(self) -> None:
        """
        Plot the confusion matrix on the test set of all the datasets in the timeline.

        Helps for comparing the different retraining periods.
        """
        n = len(self)
        n_rows = math.ceil(n / 2)
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.2, hspace=0.8)
        for i in range(n):
            row = i // 2
            col = i % 2

            (
                dataset_name,
                _,
                _,
                X_test,
                _,
                _,
                y_test,
            ) = self.datasets[i]
            period = re.findall(r"\d+", dataset_name)[1:]
            self.model_trainer.plot_confusion_matrix(
                X_test,
                y_test,
                title=f"confusion matrix: {str(period)}",
                ax=axes[row][col],
            )
            if i == self.step:
                axes[row][col].patch.set_linewidth(10)
                axes[row][col].patch.set_edgecolor("green")
                extra_text = "(initial trianing)" if i == 0 else ""
                fig.suptitle(f"Results after training on {dataset_name} {extra_text}")

        fig.tight_layout()
        fig.savefig(
            f"visualizations/bi_lstm_confusion_matrix{self.get_current_period()}"
        )

    def __len__(self):
        """Get the length of the retrainer, corresponds to the number of datasets."""
        return len(self.datasets)


def main() -> None:
    """
    Entry point for the retrainer script.

    Specifies the initial training dataset and divides the remaining years into their own datasets, runs retraining logic.
    """
    config = {
        "dataset": DatasetUsed.DEFAULT_5K_TILL_2018,
        "scaler": StandardScaler(),
        "imbalance_handling": ImbalanceHandling.RANDOM_OVER_SAMPLING,
        "sequence_length": 13,
        "padding_value": -100.0,
        "batch_size": 32,
        "test_size": 0.3,
        "validation_size": 0.2,
        "lr": 0.0001,
        "metrics": [MacroF1(masked_class=2)],
        "epochs": 50,
    }

    retrainer = Retrainer(DeepLearningModelType.BI_LSTM, config, lookback_window=5)
    retrainer.train_current_dataset()
    retrainer.plot_confusion_for_all()

    for i in range(1, len(retrainer)):
        retrainer.advance()
        retrainer.train_current_dataset()
        retrainer.plot_confusion_for_all()


if __name__ == "__main__":
    main()
