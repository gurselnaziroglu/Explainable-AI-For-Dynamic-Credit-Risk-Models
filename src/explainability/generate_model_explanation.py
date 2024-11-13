from src.constants import DatasetUsed
from sklearn.preprocessing import StandardScaler
from src.explainability.model_explainer import ModelExplainer


def main():
    config = {
        "dataset": "credit_risk_dataset_5k_processed[2006,2018].csv",   #Change dataset to be used for explanation
        "scaler": StandardScaler(),
        "imbalance_handling": None,
        "sequence_length": 17,
        "padding_value": -100.0,
        "test_size": 0,
        "validation_size": 0,
    }
    explainer_config = {
        "rs": 42,
        "nsamples": 3200,
        "top_x_events": 3,
        "top_feats": 15,
        "top_x_feats": 3,
    }
    models = [
        "bi_lstm[2006, 2018]",
        "bi_lstm[2015, 2019]",
        "bi_lstm[2016, 2020]",
        "bi_lstm[2017, 2021]",
        "bi_lstm[2018, 2022]",
    ]
    for model_name in models:
        dataset_name = config["dataset"].removesuffix(".csv")
        explainer = ModelExplainer(
            config, "final_models/" + model_name, explainer_config
        )
        filename = model_name + "-" + dataset_name
        print(filename)
        explainer.generate_shap_df(filename=filename)


if __name__ == "__main__":
    main()
