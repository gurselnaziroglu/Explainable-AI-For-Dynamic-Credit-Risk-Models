from src.models.reinforcement_learning.ReinforcementLearningBase import ReinforcementLearning
import numpy as np


class ExpectedSARSA(ReinforcementLearning):
    def update_q_table(self, q_table_pre_setter, discount_factor, learning_rate, q_table, state, action, reward,
                       next_state, preSetter, exploration_prob=None, num_actions=None, next_action=None):
        current_q = q_table[state, action]
        expected_next_q = np.sum(q_table[next_state, :] * (1 - exploration_prob) + exploration_prob / num_actions)
        new_q = current_q + learning_rate * (reward + discount_factor * expected_next_q - current_q)
        q_table[state, action] = new_q
        q_table_pre_setter[state][action] = preSetter