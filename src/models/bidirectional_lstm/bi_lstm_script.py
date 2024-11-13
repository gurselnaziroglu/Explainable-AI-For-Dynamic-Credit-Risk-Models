"""Sctipt for defining a simple bidirectional LSTM."""

from keras.layers import LSTM, Bidirectional, Dense, Masking, BatchNormalization
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential

from src.constants import DatasetUsed, ImbalanceHandling
from src.model_trainer import ModelTrainer
from src.helpers import MacroF1
import matplotlib.pyplot as plt


def get_model(padding_value: float):
    """
    Get the Bi-LSTM tensorflow model.

    :param padding_value: the value used in padding for the data. Needed for masking
    :type padding_value: float

    ...
    :return: the tensorflow model
    """
    input_shape = (None, 33)

    # model architecture
    model = Sequential()
    model.add(Masking(mask_value=padding_value, input_shape=input_shape))
    model.add(
        Bidirectional(
            LSTM(
                32,
                input_shape=input_shape,
                activation="relu",
                return_sequences=True,
            )
        )
    )
    model.add(BatchNormalization())
    model.add(Dense(1, activation="sigmoid"))

    return model


def main():
    """Train and test model.

    This Gives you an entry point to test the Bi-LSTM Model.
    Change the config for different results
    """
    # Step 1: Define your configuration dictionary
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

    # Step 2: Construct a model (Optional if you have one saved)
    model = get_model(config["padding_value"])

    # Step 3: Use the model trainer for easier training and evaluation
    # given either a model or model name to load, it loads the model and prepares the dataset from the config
    model_trainer = ModelTrainer(model, config)
    # model_trainer = ModelTrainer("bi_lstm_2006_2018", config)

    # Step 4: Simply call .train() to train your model based on the config
    model_trainer.train()

    # Step 5: .save() saves your trained model to the "models/" directory given a model name
    model_trainer.save("bi_lstm_2006_2018")

    # You can use .load() to swap the current model with another one without making a new ModelTrainer
    # model_trainer.load("bi_lstm_2006_2018")

    # Step 6: Evaluate your model

    # You can plot the confusion matrix. Training and test sets are saved in the ModelTrainer

    # model_trainer.plot_confusion_matrix(
    #     model_trainer.X_train,
    #     model_trainer.y_train,
    #     title="Training Set Confusion Matrix",
    # )

    # model_trainer.plot_confusion_matrix(
    #     model_trainer.X_val,
    #     model_trainer.y_val,
    #     title="Validation Set Confusion Matrix",
    # )

    model_trainer.plot_confusion_matrix(
        model_trainer.X_test,
        model_trainer.y_test,
        title="Historical (2018) Test Set Confusion Matrix",
    )

    # You can plot the calibration curve
    # model_trainer.plot_calibration_curve(
    #     model_trainer.X_test,
    #     model_trainer.y_test,
    #     title="Historical Test Set Calibration Curve",
    # )

    # You can plot ROC curve too!
    # model_trainer.plot_ROC_curve(
    #     model_trainer.X_test,
    #     model_trainer.y_test,
    #     title="Historical Test Set ROC Curve",
    # )

    # print_test_classification_report() prints precision, recall, f1-score, and accuracies for each class
    model_trainer.print_test_classification_report(
        model_trainer.X_test, model_trainer.y_test
    )

    _, _, X_test, _, _, y_test = model_trainer.prepare_dataset(
        "credit_risk_dataset_5k_processed[2015,2019].csv",
        val_size=0.0,
        test_size=0.3,
        fit_scaler=False,
    )

    model_trainer.plot_confusion_matrix(
        X_test,
        y_test,
        title="Future (2019) Test Set Confusion Matrix",
    )

    model_trainer.print_test_classification_report(X_test, y_test)

    plt.show()


if __name__ == "__main__":
    main()
