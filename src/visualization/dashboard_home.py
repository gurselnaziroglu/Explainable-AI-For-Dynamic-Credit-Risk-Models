import streamlit as st

st.set_page_config(layout="centered", page_title="Home")

st.title("Welcome to Explainable AI for Dynamic Credit Risk Models Dashboard!")

st.markdown("""
    The objective of this dashboard to show the results and explanations from applying Explainable AI techniques 
    to the credit risk models. It consists of 2 parts: Model Overview and Local Explanation.
""")

st.subheader("1. Model Overview")

st.markdown("""
    Model Overview page contains global explanations of different model versions using average SHAP values.
    User can compare 2 model versions to see the difference in model features attribution and events attribution.
    When the model version and dataset are selected, the dashboard will look up precomputed SHAP values from 
    pre-generated files to create the plots. Computing SHAP values for big datasets can take a long time, therefore
    the values have to be pre-generated before showing on dashboard.
    \n\n
    **How to Interpret**
    1. **Global Feature Attribution plot**  
    The plot shows SHAP values distribution for top 15 features with highest absolute mean SHAP of the selected model 
    predictions on selected dataset. Blue dots are individual values while red dits are mean value of feature.
    The top features are the main driver of probability of defaults, contributing in positive or negative direction.  
    Positive SHAP value means the presence of the feature value (when 
    not uninformative average value) increases the prediction value.  
    Negative SHAP value means the presence of the feature value decreases the prediction value.
    2. **Global Event Attribution plot**  
    The plot shows SHAP values distribution for each event (year of data) in sequence of the selected model predictions 
    on selected dataset. Index -1 means the latest year in sequence and lower index means older year. Green dots 
    are individual values and red dots are mean value of event.  
    The plot shows how impactful are data from older years, which can be useful in deciding where to
    reduce the sequence length and save computation cost.
""")

st.subheader("2. Local Explanation")

st.markdown("""
    Local Explanation page provides SHAP explanation for a single instance (client ID). User can select model version
    and dataset used for explanation, then select a client ID from the dataset to explain event-level, feature-level,
    and cell-level attribution. If the instance predicted probability of default is less than 0.5, then the
    nearest-neighbor explanation is also calculated for the instance.  
    (Please note that SHAP values may be very low for all events for features
    \n\n
    **How to Interpret**
    1. **Event-level Explanation**  
    The heatmap shows SHAP value of each event (year of data) in sequence of the selected client ID. 
    Index -1 means the latest year in sequence and lower index means older year.  
    Positive SHAP value means the presence of the event (when not uninformative average values) 
    increases the prediction value.  
    Negative SHAP value means the presence of the event decreases the prediction value.  
    2. **Feature-level Explanation**  
    The bar plot shows SHAP value of top 15 features with highest absolute SHAP value.  
    Positive SHAP value means the presence of the feature value (when 
    not uninformative average value) increases the prediction value.  
    Negative SHAP value means the presence of the feature value decreases the prediction value.  
    3. **Cell-level Explanation**  
    The heatmap shows the contribution of top 3 features at top 3 events. Each cell represents the SHAP value of a
    feature at an event in the sequence.  
    4. **Nearest Neighbor Explanation**  
    When the predicted value of the instance is lower than 0.5 (considered non-default), the algorithm calculates the
    instance distance with other instances to find the nearest neighbor that has prediction value >= 0.5 (considered 
    default). The algorithm then outputs which features from the selected instance are different from the nearest 
    default instance.
""")