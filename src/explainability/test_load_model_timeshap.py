from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential

from src.model_trainer import ModelTrainer
from src.constants import DatasetUsed, ImbalanceHandling
from src.data.transformations import sequence_train_test_split

from timeshap.utils import calc_avg_event
from timeshap.explainer import local_report
from timeshap.plot import plot_temp_coalition_pruning, plot_event_heatmap, plot_feat_barplot, plot_cell_level
from timeshap.explainer import local_pruning, local_event, local_feat, local_cell_level
import pandas as pd
import numpy as np
import altair as alt

def main():
    config = {
        "dataset": DatasetUsed.DEFAULT_5K_TILL_2018,
        "scaler": StandardScaler(),
        "imbalance_handling": ImbalanceHandling.RANDOM_OVER_SAMPLING,
        "sequence_length": 17,
        "padding_value": -100.0,
        "batch_size": 16,
        "test_size": 0.3,
        "validation_size": 0.2,
        "lr": 0.0001,
        "metrics": ["accuracy"],
        "epochs": 15,
    }

    # Step 2: Construct a model (Optional if you have one saved)
    input_shape = (config["sequence_length"], 33)

    model = Sequential()

    model_trainer = ModelTrainer(model, config)

    model_trainer.load("bi_lstm_2006_2018")
    #
    # model_trainer.plot_confusion_matrix(
    #     model_trainer.X_test, model_trainer.y_test, title="Test Set Confusion Matrix"
    # )

    # model_trainer.print_test_classification_report(
    #     model_trainer.X_test, model_trainer.y_test
    # )

    _, _, X_test, _, _, y_test = model_trainer.prepare_dataset(
        DatasetUsed.DEFAULT_5K_POST_2018, val_size=0, test_size=1.0, fit_scaler=False, load=False
    )

    # Here we are evaluating the model on our new dataset (future data)
    # model_trainer.plot_confusion_matrix(
    #     X_test, y_test, title="Post 2018 Confusion Matrix"
    # )

    # model_trainer.plot_proba_distribution(
    #     model_trainer.X_test, model_trainer.y_test, title="Predicted Probability Distribution from test set"
    # )
    #
    # model_trainer.plot_proba_distribution(
    #     X_test, y_test, title="Predicted Probability Distribution from post 2018"
    # )

    dataset_pre = pd.read_csv('data/processed/credit_risk_dataset_5k_processed[2006,2018].csv')
    feature_cols = dataset_pre.drop(columns=["Default Flag", "time_series", "id"]).columns.to_list()
    plot_feats = dict(zip(feature_cols, feature_cols))

    # def f(X):
    #     y_probas = np.round(model_trainer.model.predict(X), 2)
    #     # print(y_probas.shape)
    #     y_proba_last = np.empty(shape=(y_probas.shape[0], 1))
    #     # print(y_proba_last.shape)
    #     # print(y_probas[0].shape)
    #     for i in range(y_probas.shape[0]):
    #         y_proba_last[i] = y_probas[i][y_probas.shape[1] - 1]
    #     return y_proba_last

    f = model_trainer.make_predictor(take_last=1)

    # average_event = pd.DataFrame({col: [-100] * 1 for col in feature_cols})

    X = dataset_pre.drop(columns=["Default Flag", "time_series"])
    y = dataset_pre[["id", "Default Flag"]]

    X_train_df, _, y_train_df, _ = sequence_train_test_split(
        X, y, id_col="id", test_size=0.3, random_seed=42
    )

    y_train_df = y_train_df.reset_index(drop=True)

    X_train, _, _, y_train, _, _ = model_trainer.prepare_dataset(
        DatasetUsed.DEFAULT_5K_POST_2018, val_size=0.2, test_size=0.3, fit_scaler=False, load=False
    )

    X_train_norm_df = pd.DataFrame(X_train.reshape(-1, X_train.shape[-1]), columns=feature_cols)
    X_train_norm_df = X_train_norm_df[X_train_norm_df['Business Relation Client'] > -100].reset_index(drop=True)
    X_train_norm_df['id'] = y_train_df['id']

    average_event = calc_avg_event(X_train_norm_df, numerical_feats=feature_cols, categorical_feats=[])

    y_probas = f(X_train)
    # print(y_probas)
    y_pred = np.round(y_probas)
    print(y_pred.sum())
    is_pos = np.where(y_pred == 1)[0]
    X_train_pos = X_train[is_pos, :]

    orig_seq_len = X_train_pos != -100
    orig_seq_len = np.sum(np.all(orig_seq_len, axis=2), axis=1)

    test_index = 4
    pos_x_data = X_train_pos[test_index:test_index+1, :]
    positive_sequence_id = X_train_norm_df['id'].unique()[test_index]
    sequence_id_feat = 'id'

    print(orig_seq_len[test_index])
    print(pos_x_data[:,:orig_seq_len[test_index],:].shape)
    print(f(pos_x_data))
    print(f(pos_x_data[:,:orig_seq_len[test_index],:]))
    print(f(pos_x_data).shape)

    # pruning_dict = {'tol': 0.025}
    # event_dict = {'rs': 42, 'nsamples': 3200}
    # feature_dict = {'rs': 42, 'nsamples': 3200, 'feature_names': feature_cols, 'plot_features': plot_feats}
    # cell_dict = {'rs': 42, 'nsamples': 3200, 'top_x_feats': 3, 'top_x_events': 3}
    # loc_report = local_report(f, pos_x_data[:,:orig_seq_len[test_index],:], pruning_dict, event_dict, feature_dict, cell_dict=cell_dict,
    #              entity_uuid=positive_sequence_id, entity_col='id', baseline=average_event)
    # loc_report.show()

    event_dict = {'rs': 42, 'nsamples': 3200}
    event_data = local_event(f, pos_x_data[:,:orig_seq_len[test_index],:], event_dict, positive_sequence_id, sequence_id_feat, average_event,
                             0)
    event_plot = plot_event_heatmap(event_data)
    event_plot.show()
    #
    feature_dict = {'rs': 42, 'nsamples': 3200, 'feature_names': feature_cols, 'plot_features': plot_feats}
    feature_data = local_feat(f, pos_x_data, feature_dict, positive_sequence_id, sequence_id_feat, average_event,
                              0)
    feature_plot = plot_feat_barplot(feature_data, feature_dict.get('top_feats'), feature_dict.get('plot_features'))
    # feature_plot.encoding.x.scale = alt.Scale(domain=[-0.3,0.3])
    # print(feature_plot.to_dict())
    feature_plot.properties(
        width=400,
        height=600
    ).show()

if __name__ == "__main__":
    main()