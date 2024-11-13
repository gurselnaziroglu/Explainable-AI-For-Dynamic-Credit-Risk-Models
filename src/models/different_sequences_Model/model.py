import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.utils import to_categorical
from src.constants import DatasetUsed, ProjectRoot


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate True Positive (TP), True Negative (TN),
    False Positive (FP), and False Negative (FN).

    Parameters:
    - y_true: NumPy array, true labels (ground truth)
    - y_pred: NumPy array, predicted labels

    Returns:
    - Tuple (TP, TN, FP, FN)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")

    # Calculate confusion matrix values
    y_pred=np.round(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP, TN, FP, FN


def calculate_classification_metrics(TP, TN, FP, FN):
    """
    Calculate Precision, Recall, Specificity, F1 Score, and Accuracy.

    Parameters:
    - TP: True Positive
    - TN: True Negative
    - FP: False Positive
    - FN: False Negative

    Returns:
    - Dictionary containing the calculated metrics
    """
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    metrics = {
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1 Score': f1_score,
        'Accuracy': accuracy
    }

    return metrics


def plot_classification_metrics(metrics):
    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.bar(labels, values, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.title('Classification Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.show()


def get_min_metric(metrics):
    """
    Get the minimum value among the classification metrics.

    Parameters:
    - metrics: Dictionary containing classification metrics

    Returns:
    - Tuple (metric_name, min_value)
    """
    min_metric_name = min(metrics, key=metrics.get)
    min_metric_value = metrics[min_metric_name]
    return min_metric_name, min_metric_value


def save_model(model, model_name):
    """
    Save a model in the models/ directory using the .keras extension.

    :param model: keras model
    :param model_name: name of the keras model
    :type model_name: str
    """
    path = os.path.join(
        ProjectRoot.ROOT_PATH,
        "models",
        "sequences",
        f"{model_name}.keras",
    )
    model.save(path)
    print(f'Model saved to "{path}"')


# Read the dataset
def BatchesGenerator(seq_size):
    path = os.path.join(
        ProjectRoot.ROOT_PATH,
        "data",
        "processed",
        DatasetUsed.DEFAULT_5K_TILL_2018
    )
    dataset = pd.read_csv(path)
    # #print(dataset.shape)

    dataset['time_series'] = pd.to_datetime(dataset['time_series'], format='%Y-%m-%d')

    dataset = dataset.dropna()

    # Assuming the column containing unique IDs is named "id"
    unique_ids_counts = dataset['id'].value_counts()

    # Count of each count
    count_of_counts = unique_ids_counts.value_counts()

    counts_values = count_of_counts.index.tolist()
    counts_counts = count_of_counts.values.tolist()
    list_values = []
    list_counts = []
    for i in range(len(counts_values)):
        if counts_values[i] > seq_size:
            list_values.append(counts_values[i])
            list_counts.append(counts_counts[i])

    ids_to_extract_default_1 = dataset[dataset['Default Flag'] == 1]['id'].unique()
    ids_to_extract_default_0 = dataset[dataset['Default Flag'] == 0]['id'].unique()
    # print(ids_to_extract_default_1, ids_to_extract_default_1.shape)
    # print(ids_to_extract_default_0, ids_to_extract_default_0.shape)
    filtered_dataset_default_1 = dataset[dataset['id'].isin(ids_to_extract_default_1)]
    filtered_dataset_default_0 = dataset[dataset['id'].isin(ids_to_extract_default_0)]
    filtered_dataset_1 = filtered_dataset_default_1.sort_values(by=['id', 'time_series'])
    filtered_dataset_0 = filtered_dataset_default_0.sort_values(by=['id', 'time_series'])
    target_ids_higher_seq_1 = []
    target_ids_higher_seq_0 = []
    unique_ids_counts_1 = filtered_dataset_1['id'].value_counts()
    unique_ids_counts_0 = filtered_dataset_0['id'].value_counts()
    for i in range(len(list_values)):
        target_ids_higher_seq_1 = target_ids_higher_seq_1 + unique_ids_counts_1[
            unique_ids_counts_1 == list_values[i]].index.tolist()
        target_ids_higher_seq_0 = target_ids_higher_seq_0 + unique_ids_counts_0[
            unique_ids_counts_0 == list_values[i]].index.tolist()

    # sequences_default_1 = dataset[dataset['id'].isin(ids_to_extract_default_1)]
    # Assuming the column containing unique IDs is named "id"
    # unique_ids_counts = dataset['id'].value_counts()
    filtered_dataset_1 = dataset[dataset['id'].isin(target_ids_higher_seq_1)]
    filtered_dataset_1 = filtered_dataset_1.sort_values(by=['id', 'time_series'])
    filtered_dataset_12 = filtered_dataset_1.groupby('id').tail(seq_size)

    # Extract the ids that appeared exactly 17 times
    ids_to_extract = unique_ids_counts[unique_ids_counts == seq_size].index

    # Filter the dataset based on the selected ids
    filtered_dataset = dataset[dataset['id'].isin(ids_to_extract)]
    filtered_dataset = pd.concat([filtered_dataset, filtered_dataset_12], ignore_index=True)

    filtered_dataset = filtered_dataset.sort_values(by=['id', 'time_series'])

    y_array = filtered_dataset[["id", "Default Flag"]]
    filtered_dataset = filtered_dataset.drop(["Default Flag", "time_series"], axis=1)

    list_of_lists = [group.values.tolist() for _, group in filtered_dataset.groupby('id')]
    y_array = [group.values.tolist() for _, group in y_array.groupby('id')]

    array_of_arrays = np.array(list_of_lists)
    y_array = np.array(y_array)

    X_train, X_test, y_train, y_test = train_test_split(array_of_arrays, y_array, test_size=0.3, random_state=42)

    y_test_second_variable = np.array([y_sequence[:, 1] for y_sequence in y_test])

    # Reshape to (153, 16, 1)
    y_test_second_variable = y_test_second_variable.reshape((y_test.shape[0], y_test.shape[1], 1))

    # #print the shape of the new array
    # #print("Shape of y_test_second_variable:", y_test_second_variable.shape)

    # Assuming y_train has shape (356, 16, 2)
    # Extract the second variable for each time step
    y_train_second_variable = np.array([y_sequence[:, 1] for y_sequence in y_train])

    # Reshape to (356, 16, 1)
    y_train_second_variable = y_train_second_variable.reshape((y_train.shape[0], y_train.shape[1], 1))

    # #print the shape of the new array
    # #print("Shape of y_train_second_variable:", y_train_second_variable.shape)

    y_train=y_train_second_variable
    y_test=y_test_second_variable
    # #print the shapes of the resulting arrays
    # print("Shape of X_train:", X_train.shape)
    # print("Shape of X_test:", X_test.shape)
    # print("Shape of y_train:", y_train.shape)
    # print("Shape of y_test:", y_test.shape)

    # Loop over elements in X_train and collect unique data types
    unique_types_X_train = set()
    for sequence in X_train:
        for timestep in sequence:
            for element in timestep:
                unique_types_X_train.add(type(element))

    # Loop over elements in X_test and collect unique data types
    unique_types_X_test = set()
    for sequence in X_test:
        for timestep in sequence:
            for element in timestep:
                unique_types_X_test.add(type(element))

    # Loop over elements in y_train and collect unique data types
    unique_types_y_train = set()
    for sequence in y_train:
        for timestep in sequence:
            for element in timestep:
                unique_types_y_train.add(type(element))

    # Loop over elements in y_test and collect unique data types
    unique_types_y_test = set()
    for sequence in y_test:
        for timestep in sequence:
            for element in timestep:
                unique_types_y_test.add(type(element))

    # #print the unique data types for each array
    # print("Unique data types in X_train:", unique_types_X_train)
    # print("Unique data types in X_test:", unique_types_X_test)
    # print("Unique data types in y_train:", unique_types_y_train)
    # print("Unique data types in y_test:", unique_types_y_test)

    # print("--------------------------------------------------------------------------------------------------------------")
    return X_train, y_train ,  X_test ,y_test


def LSTM_Model(activationFunction, unitsLSTM, outputLayerActivationFunction, optimizerLSTM, lossFunction, epochs,
               batch_size, sequences, save=False, saving_threshold=[0.0]):

    print("LSTM is in progress with save mode {0} and threshold {1}...".format(save, saving_threshold))
    feature_dim = 34
    num_classes = 2
    TPTotal= 0
    TNTotal= 0
    FPTotal= 0
    FNTotal= 0
    models = {}
    for sequence_length in sequences:

        X_train, y_train, X_test, y_test = BatchesGenerator(sequence_length)

        # Define the model
        model = Sequential()
        model.add(LSTM(unitsLSTM, input_shape=(sequence_length, feature_dim), activation=activationFunction, return_sequences=True))
        model.add(Dense(num_classes, activation=outputLayerActivationFunction))

        # Compile the model
        model.compile(optimizer=optimizerLSTM, loss=lossFunction, metrics=['accuracy'])

        # #print the model summary
        # model.summary()

        # Assuming y_train and y_test are initially of shape (batch_size, sequence_length)
        num_classes = 2  # Adjust based on your problem
        y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
        y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

        # Train the model
        model.fit(X_train, y_train_one_hot, epochs=epochs, batch_size=batch_size, validation_data=(X_train, y_train_one_hot))

        models[sequence_length] = model

        y_pred = model.predict(X_test,verbose=0)
        y_true = y_test_one_hot
        TP, TN, FP, FN = calculate_confusion_matrix(y_true, y_pred)
        TPTotal = TPTotal + TP

        TNTotal = TNTotal + TN
        FPTotal = FPTotal + FP
        FNTotal = FNTotal + FN

    metrics = calculate_classification_metrics(TPTotal, TNTotal, FPTotal, FNTotal)
    min_metric_name, min_metric_value = get_min_metric(metrics)

    # save models only if the averaged metric is higher than the given threshold
    if save and min_metric_value > saving_threshold[0]:
        print("\nSaving models for different sequence lengths. saving_threshold: {0} Accuracy: {1}".format(saving_threshold[0], min_metric_value))
        saving_threshold[0] = min_metric_value
        for sequence_length, model in models.items():
            save_model(model, "sequence_{0}".format(sequence_length))
        print("\n")

    return min_metric_value
