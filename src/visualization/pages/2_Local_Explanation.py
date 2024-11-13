import streamlit as st

from sklearn.preprocessing import StandardScaler
from src.constants import DatasetUsed
from src.explainability.model_explainer import ModelExplainer
import pandas as pd
import numpy as np

from timeshap.plot import plot_event_heatmap
from src.visualization.plots import plot_feat_barplot, plot_cell_level
from src.explainability.nn_explainability import Explain
import os


def main():
    st.set_page_config(layout="wide", page_title="Local Explanation")
    st.title('Local Explainability')

    path = os.path.join(
        "models/final_models/",
    )
    models = [modelfile for modelfile in os.listdir(path) if modelfile.endswith(".h5")]
    model_filename = st.selectbox("Select model version", models)
    model_name = "final_models/" + model_filename.removesuffix(".h5")
    # st.write(model_name)

    data_path = os.path.join(
        "data/processed/",
    )
    datasets = [filename for filename in os.listdir(data_path) if filename.endswith(".csv")]
    dataset_name = st.selectbox("Select dataset", datasets)


    config = {
        "dataset": dataset_name,  # Change dataset to be used for explanation
        "scaler": StandardScaler(),
        "imbalance_handling": None,
        "sequence_length": 17,
        "padding_value": -100.0,
        "test_size": 0,
        "validation_size": 0,
    }
    explainer_config = {
        'rs': 42,
        'nsamples': 3200,
        'top_feats': 15,
        'top_x_events': 3,
        'top_x_feats': 3,
    }
    with st.spinner('Please wait...'):
        explainer = ModelExplainer(config, model_name, explainer_config)
        dataset = explainer.dataset

        #Make selectbox for customer ID to show explanation
        ids = dataset['id'].unique()

    # st.write('Select customer ID to see local explanations')
    id = st.selectbox('Select customer ID to see local explanations', ids)


    if (id != None):

        st.dataframe(dataset[dataset['id'] == id])
        index = np.where(ids == id)[0][0]
        index = int(index)

        x = explainer.X_train[index:index+1, :, :]

        # st.dataframe(explainer.model.predict(x)[0])
        predicted = explainer.predictor(x)[0][0]
        st.write('Predicted value = {:.4f}'.format(predicted))

        col1, col2 = st.columns(2)

        with col1:
            tab1, tab2, tab3 = st.tabs(["Event level","Feature level", "Cell level"])

            event_data = explainer.get_event_data(x)
            event_plot = plot_event_heatmap(event_data).properties(width=300, height=400)
            tab1.subheader("Event-level Contribution")
            tab1.altair_chart(event_plot)

            feature_data = explainer.get_feature_data(x)
            feature_dict = explainer.feature_dict
            feature_plot = plot_feat_barplot(feature_data, feature_dict.get('top_feats'),
                                             feature_dict.get('plot_features')).properties(width=600, height=400)
            tab2.subheader("Feature-level Contribution for Top {:.0f} Features".format(feature_dict.get('top_feats')))
            tab2.altair_chart(feature_plot)
            #
            cell_plot_params = {
                # 'axis_lim': [-0.6, 0.6],
                'FontSize': 10,
                'height': 380,
                'width': 800
            }
            cell_data = explainer.get_cell_data(x, event_data, feature_data)
            feat_names = list(feature_data['Feature'].values)[:-1] # exclude pruned events
            cell_plot = plot_cell_level(cell_data, feat_names, feature_dict.get('plot_features'),cell_plot_params)#.properties(width=600,height=280)
            tab3.subheader("Cell-level Contribution")
            tab3.altair_chart(cell_plot)

        with col2:
            if predicted < 0.5:
                st.subheader("Explanation with Nearest-Neighbor")
                with st.spinner("Calculating..."):
                    nn_explain = Explain(id, dataset)
                st.text(nn_explain)


if __name__ == "__main__":
    main()