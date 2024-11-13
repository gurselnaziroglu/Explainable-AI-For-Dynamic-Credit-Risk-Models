import streamlit as st
from timeshap.plot import plot_global_feat, plot_global_event
import pandas as pd
import altair as alt
import os
import re

def main():
    st.set_page_config(layout="wide", page_title="Model Overview")
    st.title('Model Overview')
    st.header("Model Explanation with  Shapley Values")

    path = os.path.join(
        "data/model_logs/event_data/",
    )
    files = os.listdir(path)
    models = sorted(list(set([string.split("-")[0] for string in files])))
    datasets = sorted(list(set([string.split("-")[1].removesuffix("_event_data.csv") for string in files])))

    col1, col2 = st.columns(2)
    model1 = col1.selectbox("Select model version", models, key="model1")
    dataset1 = col1.selectbox("Select dataset version", datasets, key="dataset1")

    model2 = col2.selectbox("Select model version", models, key="model2")
    dataset2 = col2.selectbox("Select dataset version", datasets, key="dataset2")


    plot_params = {
        'axis_lim': [-0.6, 0.6],
        'FontSize': 10,
        'width': 400
    }
    event_plot_params = {
        # 'axis_lim': [-0.6, 0.6],
        # 'FontSize': 10,
        'width': 500
    }

    with st.spinner("Loading..."):

        feature_filepath_1 = os.path.join(f"data/model_logs/feature_data/{model1}-{dataset1}_feature_data.csv")
        feature_df_1 = pd.read_csv(feature_filepath_1)
        global_feat_plot_1 = plot_global_feat(feature_df_1[feature_df_1['prediction'] >= 0.5], 15, None, None,plot_params)
        alt.data_transformers.disable_max_rows()

        feature_filepath_2 = os.path.join(f"data/model_logs/feature_data/{model2}-{dataset2}_feature_data.csv")
        feature_df_2 = pd.read_csv(feature_filepath_2)
        global_feat_plot_2 = plot_global_feat(feature_df_2[feature_df_2['prediction'] >= 0.5], 15, None, None,plot_params)

        event_filepath_1 = os.path.join(f"data/model_logs/event_data/{model1}-{dataset1}_event_data.csv")
        event_df_1 = pd.read_csv(event_filepath_1)
        # event_df_1["t (event index)"] = event_df_1["Feature"].apply(
        #     lambda x: 1 if x == 'Pruned Events' else -int(re.findall(r'\d+', x)[0]) + 1)
        global_event_plot_1 = plot_global_event(event_df_1[event_df_1['prediction'] >= 0.5], event_plot_params)

        event_filepath_2 = os.path.join(f"data/model_logs/event_data/{model2}-{dataset2}_event_data.csv")
        event_df_2 = pd.read_csv(event_filepath_2)
        global_event_plot_2 = plot_global_event(event_df_2[event_df_2['prediction'] >= 0.5], event_plot_params)


    col1.subheader(f"Feature Attribution of Model: {model1}")
    col1.altair_chart(global_feat_plot_1, use_container_width=True)
    col1.subheader(f"Event Attribution of Model: {model1}")
    col1.altair_chart(global_event_plot_1, use_container_width=True)

    col2.subheader(f"Feature Attribution of Model: {model2}")
    col2.altair_chart(global_feat_plot_2, use_container_width=True)
    col2.subheader(f"Event Attribution of Model: {model2}")
    col2.altair_chart(global_event_plot_2, use_container_width=True)



if __name__ == "__main__":
    main()

