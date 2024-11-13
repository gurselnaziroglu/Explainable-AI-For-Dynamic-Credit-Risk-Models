"""Contains the logic of the Bi-LSTM evaluation function given to the RL."""

import os
from keras.layers import LSTM, Bidirectional
from keras import backend as K
from tensorflow.keras.models import Sequential

from src.model_trainer import ModelTrainer
from src.helpers import MacroF1
from src.data.process_dataset import process_dataset_call
from copy import deepcopy

# constants
RL_DATASET_PREFIX = "credit_risk_dataset_5k"

# Global variables
_current_model: ModelTrainer | None = None
_eval_retraining_period: list[int] | None = None


def set_evaluation_config(
    current_model: ModelTrainer, eval_retraining_period: list[int]
) -> None:
    """Set the global variables.

    Unfortunately, Global variables exist here because of how inconveniently RL and GA are coded.
    """
    global _current_model, _eval_retraining_period

    _current_model = current_model
    _eval_retraining_period = eval_retraining_period


def modify_model(model, lstm_activation: str, optimizer: str, config: dict):
    """
    Change model optimizer, and LSTM layer activation.

    :param model: model to be modified

    :param lstm_activation: lstm layer activation function
    :type lstm_activation: str

    :param optimizer: name of the new optimizer.
    :type optimizer: str
    ...
    :return: the modified model
    """
    existing_lstm_layer = model.layers[1]

    new_lstm_layer = Bidirectional(
        LSTM(
            units=existing_lstm_layer.layer.units,
            input_shape=existing_lstm_layer.input_shape,
            activation=lstm_activation,
            return_sequences=True,
        )
    )

    new_model = Sequential()
    for i, layer in enumerate(model.layers):
        if i == 1:
            new_model.add(new_lstm_layer)
        else:
            new_model.add(layer)

    new_model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=config["metrics"],
    )

    K.set_value(model.optimizer.learning_rate, config["lr"])
    new_model.set_weights(model.get_weights())

    return new_model


def evaluate_bi_lstm(
    activationFunction: str,
    optimizerLSTM: str,
    epochs: int,
    batch_size: int,
    saving_threshold: list[int] = [0.0],
) -> float:
    """
    Eavluate the Bi-LSTM model.

    Should only be invoked when the global variables are set.
    Meaning we have a model trainer with the old model, and we have
    the retrianing period to tune the old model on.

    :param activationFunction: name of the new activation function of the LSTM layer.
    :type activationFunction: str


    :param optimizerLSTM: name of the new optimizer.
    :type optimizerLSTM: str


    :param epochs: number of epochs to train/tune on.
    :type epochs: int

    :param batch_size: batch size for trianing/tuning.
    :type batch_size: int

    ...
    :return: the macro-F1 score evaluated on the validation dataset of this particular period.
    :rtype: float
    """
    assert _eval_retraining_period is not None, "Training period is unknown"
    assert _current_model is not None, "No existing model to run RL on"
    trainer = deepcopy(_current_model)

    start, end = _eval_retraining_period
    processed_path = f"data/processed/{RL_DATASET_PREFIX}_processed[{start},{end}].csv"
    if not os.path.isfile(processed_path):
        processed_path = process_dataset_call(
            src="data/raw",
            dest="data/processed",
            start=start,
            end=end,
            dataset_file_name=f"{RL_DATASET_PREFIX}.csv",
        )[0]

    retraining_dataset_name = os.path.basename(processed_path)

    config = {
        "dataset": retraining_dataset_name,
        "sequence_length": 13,  # maximum sequence length
        "batch_size": batch_size,
        "test_size": 0.3,
        "validation_size": 0.2,
        "lr": 0.0001,
        "metrics": [MacroF1(masked_class=2)],
        "epochs": epochs,
    }

    trainer.prepare_dataset(
        config["dataset"],
        val_size=config["validation_size"],
        test_size=config["test_size"],
        fit_scaler=False,
        load=True,
    )
    trainer.config = trainer.config | config
    trainer.model = modify_model(
        trainer.model, activationFunction, optimizerLSTM, config
    )

    macro_f1 = trainer.train(plot_loss=False)

    if macro_f1 > saving_threshold[0]:
        trainer.save(f"final_models/bi_lstm{_eval_retraining_period}")

    return macro_f1
