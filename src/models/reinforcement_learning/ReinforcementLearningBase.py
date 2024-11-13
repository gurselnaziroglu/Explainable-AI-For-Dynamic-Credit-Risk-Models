from abc import ABC, abstractmethod

import numpy as np
from src.models.different_sequences_Model import model
from src.constants import DeepLearningModelType
from src.models.bidirectional_lstm.bi_lstm_rl_evaluation import evaluate_bi_lstm


def PreSetterModifier(
    oldPersetter,
    activationFunction,
    unitsLSTM,
    outputLayerActivationFunction,
    optimizerLSTM,
    lossFunction,
    epochs,
    batch_size,
    sequences_to_train,
    action,
):
    print("value before action {0} is {1}".format(action, oldPersetter[action]))
    newPresetter = oldPersetter.copy()
    # Check if action is 0
    if action == 0:
        print("the action changes activationFunction...")
        if oldPersetter[action] >= len(activationFunction) - 1:
            newPresetter[action] = 0
        else:
            newPresetter[action] += 1

    # Check if action is 1
    elif action == 1:
        print("the action changes unitsLSTM...")
        if oldPersetter[action] >= len(unitsLSTM) - 1:
            newPresetter[action] = 0
        else:
            newPresetter[action] += 1

    # Check if action is 2
    elif action == 2:
        print("the action changes outputLayerActivationFunction...")
        if oldPersetter[action] >= len(outputLayerActivationFunction) - 1:
            newPresetter[action] = 0
        else:
            newPresetter[action] += 1

    # Check if action is 3
    elif action == 3:
        print("the action changes optimizerLSTM...")
        if oldPersetter[action] >= len(optimizerLSTM) - 1:
            newPresetter[action] = 0
        else:
            newPresetter[action] += 1

    # Check if action is 4
    elif action == 4:
        print("the action changes lossFunction...")
        if oldPersetter[action] >= len(lossFunction) - 1:
            newPresetter[action] = 0
        else:
            newPresetter[action] += 1

    # Check if action is 5
    elif action == 5:
        print("the action changes epochs...")
        if oldPersetter[action] >= len(epochs) - 1:
            newPresetter[action] = 0
        else:
            newPresetter[action] += 1

    # Check if action is 6
    elif action == 6:
        print("the action changes batch_size...")
        if oldPersetter[action] >= len(batch_size) - 1:
            newPresetter[action] = 0
        else:
            newPresetter[action] += 1

    # Check if action is 7
    elif action == 7:
        print("the action changes sequences_to_train...")
        if oldPersetter[action] >= len(sequences_to_train) - 1:
            newPresetter[action] = 0
        else:
            newPresetter[action] += 1
    print("value after action {0} is {1}".format(action, newPresetter[action]))

    return newPresetter


def findListIndex(list1, lists):
    for index, list2 in enumerate(lists):
        if are_lists_equal(list1, list2):
            return index
    return -1


def isListInside(list1, lists):
    for list2 in lists:
        if list1 == list2:
            return True
    return False


def are_lists_equal(list1, list2):
    if len(list1) != len(list2):
        return False

    for element1, element2 in zip(list1, list2):
        if element1 != element2:
            return False

    return True


def PerformAction(
    preSetters,
    activationFunction,
    unitsLSTM,
    outputLayerActivationFunction,
    optimizerLSTM,
    lossFunction,
    epochs,
    batch_size,
    sequences_to_train,
    preSettersHistory,
    preSettersValue,
    best_fitness,
    deep_learning_model_type,
):
    if isListInside(preSetters, preSettersHistory):
        print(
            "model with preSetters {0} progressed before so skipping".format(preSetters)
        )
        return preSettersValue[findListIndex(preSetters, preSettersHistory)]
    else:
        evaluation_metric = 0.0
        if deep_learning_model_type == DeepLearningModelType.DIFFERENT_SEQUENCES_LSTM:
            evaluation_metric = model.LSTM_Model(
                activationFunction[preSetters[0]],
                unitsLSTM[preSetters[1]],
                outputLayerActivationFunction[preSetters[2]],
                optimizerLSTM[preSetters[3]],
                lossFunction[preSetters[4]],
                epochs[preSetters[5]],
                batch_size[preSetters[6]],
                sequences_to_train,
                True,
                best_fitness,
            )
        elif deep_learning_model_type == DeepLearningModelType.BI_LSTM:
            evaluation_metric = evaluate_bi_lstm(
                activationFunction[preSetters[0]],
                optimizerLSTM[preSetters[3]],
                epochs[preSetters[5]],
                batch_size[preSetters[6]],
                best_fitness,
            )
        else:
            ValueError("Unsupported model type:", deep_learning_model_type)

        return int(evaluation_metric * 100)


def select_action(exploration_prob, num_actions, q_table, state):
    if np.random.rand() < exploration_prob:
        # Exploration: choose a random action
        return np.random.choice(num_actions)
    else:
        # Exploitation: choose the action with the highest Q-value for the current state
        return np.argmax(q_table[state, :])


class ReinforcementLearning(ABC):
    def __init__(
        self,
        preSetters=None,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_prob=0.5,
        best_fitness=[0.0],
        deep_learning_model_type=DeepLearningModelType.DIFFERENT_SEQUENCES_LSTM,
    ):
        self.preSetters = preSetters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.best_fitness = best_fitness
        self.deep_learning_model_type = deep_learning_model_type

    def run(self):
        if self.preSetters is None:
            preSetters = [1, 1, 1, 1, 1, 1, 1]

        num_states = 101
        num_actions = 7
        q_table = np.zeros((num_states, num_actions))
        q_table_presetter = [
            [None for _ in range(num_actions)] for _ in range(num_states)
        ]

        num_episodes = 3
        current_state = 0
        next_state = 0
        activationFunction = ["relu", "sigmoid", "tanh"]
        unitsLSTM = [32, 64, 128, 256]
        outputLayerActivationFunction = ["softmax", "sigmoid"]
        optimizerLSTM = ["adam", "sgd", "rmsprop", "adagrad"]
        lossFunction = [
            "categorical_crossentropy",
            "binary_crossentropy",
            "mean_squared_error",
        ]
        epochs = [5, 10, 20, 50]
        batch_size = [32, 64, 128]
        sequences_to_train = list(
            range(5, 14)
        )

        preSettersHistory = []
        predictionHistory = []
        for episode in range(num_episodes):
            print("episode {0} is in progress...".format(episode))

            # Start state
            action = select_action(
                self.exploration_prob, num_actions, q_table, next_state
            )

            preSetters = PreSetterModifier(
                self.preSetters,
                activationFunction,
                unitsLSTM,
                outputLayerActivationFunction,
                optimizerLSTM,
                lossFunction,
                epochs,
                batch_size,
                sequences_to_train,
                action,
            )
            accuracy = PerformAction(
                preSetters,
                activationFunction,
                unitsLSTM,
                outputLayerActivationFunction,
                optimizerLSTM,
                lossFunction,
                epochs,
                batch_size,
                sequences_to_train,
                preSettersHistory,
                predictionHistory,
                self.best_fitness,
                self.deep_learning_model_type,
            )
            # print(accuracy)
            predictionHistory.append(accuracy)
            preSettersHistory.append(preSetters)
            print("predictionHistory:{0}".format(predictionHistory))
            print("preSettersHistory:{0}".format(preSettersHistory))

            reward = accuracy - current_state  # Use accuracy as the reward
            current_state = (
                next_state  # Move to the next state, limited to 100 as finish state
            )

            next_state = accuracy

            # this is only needed for SARSA
            next_action = select_action(
                self.exploration_prob, num_actions, q_table, next_state
            )

            # Update the Q-table
            self.update_q_table(
                q_table_presetter,
                self.discount_factor,
                self.learning_rate,
                q_table,
                current_state,
                action,
                reward,
                next_state,
                preSetters,
                self.exploration_prob,
                num_actions,
                next_action,
            )
            if next_state >= num_states - 1:
                print(f"Reached finish state 100 in episode {episode + 1}")
                return (
                    100,
                    preSettersHistory[predictionHistory.index(max(predictionHistory))],
                )

        else:
            return (
                max(predictionHistory),
                preSettersHistory[predictionHistory.index(max(predictionHistory))],
            )

    @abstractmethod
    def update_q_table(
        self,
        q_table_pre_setter,
        discount_factor,
        learning_rate,
        q_table,
        state,
        action,
        reward,
        next_state,
        preSetter,
        exploration_prob=None,
        num_actions=None,
        next_action=None,
    ):
        # Must be implemented by each Reinforcement Algorithm as they all have different update rules.
        pass
