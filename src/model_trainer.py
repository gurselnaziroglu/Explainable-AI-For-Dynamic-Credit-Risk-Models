"""Module containing the ModelTrainer class code."""

import os

import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
import seaborn as sns
from keras.models import load_model
from numpy.typing import NDArray
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam

from src.data.load_data import load_default_dataset
from src.data.transformations import pad_sequences
from src.helpers import get_class_weights, y_remove_padding, print_separator
from src.constants import ImbalanceHandling
from sklearn.calibration import CalibrationDisplay

from src.helpers import take_last_n_from_array


class ModelTrainer:
    """This class makes training models easier on different datasets.

    It helps with loading the desired dataset, preparing it for your model,
    compiling, taining, as well as evaluating the model.

    :param model: the model that will be used. No need to specify an optimizer or a loss function. They will be set to Adam and binary_crossentropy.
    A model name can be given to load the model from storage.
    :type model: any

    :param config: experiment configuration.
    :type config: dict
    """

    def __init__(self, model, config: dict):
        """Construct an instance."""
        self.config = config

        if isinstance(model, str):
            self.load(model)
        else:
            self.model = model

            opt = Adam(learning_rate=self.config["lr"])
            self.model.compile(
                optimizer=opt,
                loss="binary_crossentropy",
                metrics=self.config["metrics"],
            )

        self.prepare_dataset(
            self.config["dataset"],
            val_size=self.config["validation_size"],
            test_size=self.config["test_size"],
            fit_scaler=True,
            load=True,
        )

    def prepare_dataset(
        self,
        dataset: str,
        *,
        val_size: float,
        test_size: float,
        fit_scaler=True,
        load=False,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Prepare a dataset for the model.

        :param dataset: the dataset to be loaded
        :type dataset: DatasetUsed


        :param test_size: the size of the validation set w.r.t the testing set [0.0,1.0].
        :type test_size: float

        :param test_size: the size of the test set [0.0,1.0].
        :type test_size: float

        :param fit_scaler: fit the given scaler.
        :type fit_scaler: bool

        :param load: load the data to the ModelTrainer instance.
        :type load: bool

        ...
        :return: X_train, X_test, y_train, y_tesT
        :rtype: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        """
        scale_excluded_columns = {
            "time_series",
            "id",
            "Covenants Breach last 12 Month",
            "DPD 10 Days Flag",
            "Lead Syndicated Flag",
            "Commercial Property",
            "Syndicated Flag",
            "Default Flag",
        }
        feature_length = 33

        X_train, X_val, X_test, y_train, y_val, y_test = load_default_dataset(
            dataset,
            val_size=val_size,
            test_size=test_size,
            scaler=self.config["scaler"],
            scale_excluded_columns=scale_excluded_columns,
            imbalance_handling=(
                self.config["imbalance_handling"]
                if self.config["imbalance_handling"] != ImbalanceHandling.CLASS_WEIGHTS
                and test_size != 1.0
                else None
            ),
            fit_scaler=fit_scaler,
        )
        if "padding_value" in self.config:
            X_train = pad_sequences(
                X_train,
                max_length=self.config["sequence_length"],
                feature_length=feature_length,
                padding_value=self.config["padding_value"],
            )

            y_train = pad_sequences(
                y_train,
                max_length=self.config["sequence_length"],
                feature_length=1,
                padding_value=2.0,
            )

            X_val = pad_sequences(
                X_val,
                max_length=self.config["sequence_length"],
                feature_length=feature_length,
                padding_value=self.config["padding_value"],
            )

            y_val = pad_sequences(
                y_val,
                max_length=self.config["sequence_length"],
                feature_length=1,
                padding_value=2.0,
            )

            X_test = pad_sequences(
                X_test,
                max_length=self.config["sequence_length"],
                feature_length=feature_length,
                padding_value=self.config["padding_value"],
            )

            y_test = pad_sequences(
                y_test,
                max_length=self.config["sequence_length"],
                feature_length=1,
                padding_value=2.0,
            )

        X_train, X_val, X_test, y_train, y_val, y_test = (
            np.array(X_train),
            np.array(X_val),
            np.array(X_test),
            np.array(y_train),
            np.array(y_val),
            np.array(y_test),
        )

        if load:
            self.assign_dataset(
                dataset,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
            )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def assign_dataset(
        self, dataset_name, X_train, X_val, X_test, y_train, y_val, y_test
    ):
        self.config["dataset"] = dataset_name
        (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        ) = (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )

    def train(self, plot_loss: bool = True) -> float:
        """Invoke the model's training function with the given configuration."""
        print_separator("Model Training")
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=int(self.config["epochs"]),
            batch_size=int(self.config["batch_size"]),
            validation_data=(self.X_val, self.y_val),
            class_weight=(
                get_class_weights(self.y_train)
                if self.config["imbalance_handling"] == ImbalanceHandling.CLASS_WEIGHTS
                else None
            ),
        )

        if plot_loss:
            print("Plotting training/val loss curve...")
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("Model Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(["Train", "Val"], loc="upper left")
            plt.show()

        return history.history.get("val_macro_f1", [0.0])[-1]

    def plot_confusion_matrix(
        self, X: NDArray[np.float64], y: NDArray[np.float64], *, title: str, ax=None
    ) -> None:
        """
        Plot the confusion matrix of a given dataset using the given model.

        :param X: inputs array.
        :type X: NDArray[np.float64]

        :param y: outputs array.
        :type y: NDArray[np.float64]

        :param title: confusion matrix plot title.
        :type title: str
        """
        if ax is None:
            _, ax = plt.subplots()

        print_separator(title)

        predictor_last = self.make_predictor(1)
        y_last = take_last_n_from_array(arr=y, padding_value=2.0, n=1)
        y_pred = np.round(predictor_last(X))

        if "padding_value" in self.config:
            y_ravel, y_pred_ravel = y_remove_padding(y_last, y_pred, 2.0)

        conf_matrix_train = confusion_matrix(y_ravel, y_pred_ravel)

        sns.heatmap(
            conf_matrix_train,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

    def plot_ROC_curve(
        self, X: NDArray[np.float64], y: NDArray[np.float64], *, title: str, ax=None
    ) -> None:
        """
        Plot the ROC curve of a given dataset using the given model.

        :param X: inputs array.
        :type X: NDArray[np.float64]

        :param y: outputs array.
        :type y: NDArray[np.float64]

        :param title: ROC curve title.
        :type title: str
        """
        if ax is None:
            _, ax = plt.subplots()
        print_separator(title)

        predictor_last = self.make_predictor(1)
        y_last = take_last_n_from_array(arr=y, padding_value=2.0, n=1)
        y_probas = predictor_last(X)
        y_probas = np.hstack((1 - y_probas, y_probas))

        skplt.metrics.plot_roc(y_last, y_probas, title=title, ax=ax)

    def plot_calibration_curve(
        self, X: NDArray[np.float64], y: NDArray[np.float64], *, title: str, ax=None
    ) -> None:
        """
        Plot the calibration curve of a given dataset using the given model.

        :param X: inputs array.
        :type X: NDArray[np.float64]

        :param y: outputs array.
        :type y: NDArray[np.float64]

        :param title: calibration curve title.
        :type title: str
        """
        if ax is None:
            _, ax = plt.subplots()

        print_separator(title)
        y_probas = self.model.predict(X)

        predictor_last = self.make_predictor(1)
        y_last = take_last_n_from_array(arr=y, padding_value=2.0, n=1)
        y_probas = predictor_last(X)

        CalibrationDisplay.from_predictions(
            y_true=y_last,
            y_prob=y_probas,
            n_bins=10,
            name="Predicted",
            color="black",
            ax=ax,
        )

        ax.set_title(title)

    def print_test_classification_report(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> None:
        """
        Print the classification report for a given dataset.

        :param X: inputs array.
        :type X: NDArray[np.float64]

        :param y: outputs array.
        :type y: NDArray[np.float64]
        """
        y_pred = np.round(self.model.predict(X))

        y_ravel, y_pred_ravel = y.ravel(), y_pred.ravel()

        if "padding_value" in self.config:
            y_ravel, y_pred_ravel = y_remove_padding(y_ravel, y_pred_ravel, 2.0)

        print_separator("Classification Report")
        print(classification_report(y_ravel, y_pred_ravel, digits=4))

    def make_predictor(self, take_last: int | None = None):
        """
        Make a predict function that only conciders the last take_last number of predictions in each sequence.

        :param take_last: number of predictions to be considered from the end of each sequence.
        None considers all outputs in the sequence.
        :type take_last: int | None

        :return: the predict function that returns probabilities shaped (#samples * take_last, 1)
        :rtype: Callable[[np.NDArray[np.float64]], np.NDArray[np.float64]]
        """
        taken_pred_num = min(
            self.config["sequence_length"],
            take_last or self.config["sequence_length"] + 1,
        )

        def predict(X: NDArray[np.float64]) -> NDArray[np.float64]:
            """
            Predict the given sequences and take the last take_last probabilities from each sequence (ignores padding).

            :param X: the (padded or not) input sequence array.
            :type X: NDArray[np.float64]

            :return: the array of predictions shaped (#samples * take_last, 1)
            :rtype: np.NDArray[np.float64]
            """
            assert (
                len(X.shape) == 3
            ), "X should have a shape of (<batch size>, <sequence length>, <batch length>)"
            single_instance = X.shape[0] == 1

            # if no padding happened, default to a nummber outside the inout range (-100).
            # this way nothing will be considered padding
            orig_seq_len = X != (
                self.config["padding_value"] if "padding_value" in self.config else -100
            )

            orig_seq_len = np.all(orig_seq_len, axis=2)
            orig_seq_len = np.sum(orig_seq_len, axis=1)

            # don't pad if we only have 1 instance
            y_probas = self.model.predict(
                X[:, : orig_seq_len[0], :] if single_instance else X
            )
            original_y_probas = [
                row[:length] for row, length in zip(y_probas, orig_seq_len)
            ]
            trimmed_y_probas = np.array(
                [row[-taken_pred_num:] for row in original_y_probas]
            )
            trimmed_y_probas_ravel = trimmed_y_probas.reshape(-1, 1)

            return trimmed_y_probas_ravel

        return predict

    def save(self, sub_path: str):
        """
        Save a model in the models/ directory using the .keras extension.

        :param sub_path: path of the model inside the models/ folder
        :type sub_path: str
        """
        path = os.path.join(
            "models",
            f"{sub_path}.h5",
        )
        self.model.save(path, save_format="h5")
        print(f'Model saved to "{path}"')

    def load(self, sub_path: str) -> None:
        """
        Load a model in the models/ directory having the .keras extension.

        :param sub_path: path of the model inside the models/ folder
        :type sub_path: str
        """
        path = os.path.join(
            "models",
            f"{sub_path}.h5",
        )
        self.model = load_model(path)
        print(f'Model loaded from "{path}"')
