# Explainable AI for Dynamic Credit Risk Models

In the dynamic and complex domain of financial credit, accurately predicting loan defaults is paramount for risk management and decision-making processes. This project introduces a sophisticated predictive model that integrates Long Short-Term Memory (LSTM) net- works with Reinforcement Learning (RL) and Genetic Algorithm (GA)s, creating a dynamic, adaptable, and highly accurate system for forecasting customer loan defaults. Unlike traditional static models, our approach continuously fine-tunes the LSTM hyperparameters through RL, ensuring the model’s sensitivity and responsiveness to emerging data trends and economic shifts. The optimization of the RL model’s hyperparameters is further refined using a GA, enhancing the model’s efficiency and effectiveness in adapting to new patterns in financial credit sequential data. A key feature of our project is the emphasis on model explainability and transparency, crucial in the financial sector where decisions such as loan application rejections require clear justification. We employ Shapley values to provide insights into the contribution of individual features to the prediction outcomes, alongside the Nearest Neighbour algorithm for comparative analysis, offering a comprehensive understanding of the factors influencing default predictions. Our approach stands out by not only addressing the need for dynamic predictive capabilities in the face of emerging data evolutions, but also by fulfilling the financial industry’s demand for transparent and Explainable AI (XAI) solutions.

PS: We implemented this project as a group of 5 master's students under the supervision of experts in the field.


## Project Organization

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources (used in preprocessing).
    │   ├── model_logs     <- data used for global explainability
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Functions and classes to process or generate data
    │   │
    │   ├── explainability <- Functions and classes needed for explaining the modesl
    │   │
    │   ├── models         <- Functions and classes define models
    │   │
    │   └── visualization  <- Functions and classes to create result-oriented/explainability visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
