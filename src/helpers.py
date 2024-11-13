"""General helper functions."""

import keras as K
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray


def get_class_weights(
    y_train: NDArray[np.float64], ignored_class: float | None = None
) -> dict[float, float]:
    """
    Get the weights of classes in the training set as 1/class_count.

    :param y_train: the array of labels of the training data.
    :type y_train: NDArray[np.float64]

    :param ignored_class: this class value will be set to 0, leave empty if you want all classes to have weights.
    :type ignored_class: float | None
    ...
    :return: class weight dictionary
    :rtype: dict[float, float]
    """
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = dict(
        zip([elem for elem in unique], [1 / count for count in counts])
    )

    if ignored_class is not None:
        class_weights[ignored_class] = 0.0

    return class_weights


def y_remove_padding(
    y: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    padding_value,
) -> NDArray[np.float64]:
    """
    Remove the padding from a response array.

    :param y: the array of labels (flattened).
    :type y: NDArray[np.float64]

    :param padding_value: padding value to be ignored.
    :type padding_value: float
    ...
    :return: label array without padding.
    :rtype: NDArray[np.float64]
    """
    masked_values = y != padding_value

    return y[masked_values], y_pred[masked_values]


def print_separator(title: str) -> None:
    """
    Print a separator similar to "====title====".

    :param title: middle text.
    :type title: str
    """
    length = 150
    filler = max(length - len(title), 0)

    left = "=" * (filler // 2)
    right = "=" * (filler - filler // 2)
    print(f"{left}{title}{right}")


@K.saving.register_keras_serializable()
class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, masked_class, name="macro_f1", **kwargs):
        super(MacroF1, self).__init__(name=name, **kwargs)
        self.masked_class = masked_class
        self.true_positives1 = self.add_weight(
            name="true_positives1", initializer="zeros"
        )
        self.true_positives0 = self.add_weight(
            name="true_positives0", initializer="zeros"
        )
        self.false_positives1 = self.add_weight(
            name="false_positives1", initializer="zeros"
        )
        self.false_negatives1 = self.add_weight(
            name="false_negatives1", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), shape=(-1,))
        y_pred = tf.math.round(tf.reshape(tf.cast(y_pred, tf.float32), shape=(-1,)))

        mask = y_true != self.masked_class
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        true_positives1 = tf.reduce_sum(y_true * y_pred)
        true_positives0 = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        false_positives1 = tf.reduce_sum((1 - y_true) * y_pred)
        # false_negatives0 = false_positives1
        false_negatives1 = tf.reduce_sum(y_true * (1 - y_pred))
        # false_positives0 = false_negatives1

        self.true_positives1.assign_add(true_positives1)
        self.true_positives0.assign_add(true_positives0)
        self.false_positives1.assign_add(false_positives1)
        self.false_negatives1.assign_add(false_negatives1)

    def result(self):
        precision0 = self.true_positives0 / (
            self.true_positives0 + self.false_negatives1 + tf.keras.backend.epsilon()
        )
        precision1 = self.true_positives1 / (
            self.true_positives1 + self.false_positives1 + tf.keras.backend.epsilon()
        )

        recall0 = self.true_positives0 / (
            self.true_positives0 + self.false_positives1 + tf.keras.backend.epsilon()
        )

        recall1 = self.true_positives1 / (
            self.true_positives1 + self.false_negatives1 + tf.keras.backend.epsilon()
        )

        f1_0 = (
            2
            * (precision0 * recall0)
            / (precision0 + recall0 + tf.keras.backend.epsilon())
        )

        f1_1 = (
            2
            * (precision1 * recall1)
            / (precision1 + recall1 + tf.keras.backend.epsilon())
        )

        macro_f1 = (f1_0 + f1_1) / 2
        return macro_f1

    def reset_state(self):
        self.true_positives0.assign(0)
        self.true_positives1.assign(0)
        self.false_positives1.assign(0)
        self.false_negatives1.assign(0)

    def get_config(self):
        return {"masked_class": self.masked_class}


def take_last_n_from_array(arr, padding_value, n: int | None = None):
    assert (
        len(arr.shape) == 3
    ), "X should have a shape of (<batch size>, <sequence length>, <batch length>)"
    orig_seq_len = arr != padding_value

    orig_seq_len = np.all(orig_seq_len, axis=2)
    orig_seq_len = np.sum(orig_seq_len, axis=1)

    taken_pred_num = min(
        arr.shape[1],
        n or arr.shape[1] + 1,
    )

    # don't pad if we only have 1 instance
    original_sequence = [row[:length] for row, length in zip(arr, orig_seq_len)]
    trimmed_sequence = np.array([row[-taken_pred_num:] for row in original_sequence])
    trimmed_sequence_ravel = trimmed_sequence.reshape(-1, 1)

    return trimmed_sequence_ravel
