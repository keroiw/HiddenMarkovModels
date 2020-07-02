import numpy as np
import os
import pandas as pd


from collections import OrderedDict
from hmmlearn import hmm


HIDDEN_STATES = [i for i in range(2, 4) if i % 2 == 0]


class ModelsContainer:
    __slots__ = ['models', 'scores', 'states']

    def __init__(self):
        self.models = []
        self.scores = []
        self.states = []

    def add_model(self, model, score, state):
        self.models.append(model)
        self.scores.append(score)
        self.states.append(state)

    def get_best_model(self):
        max_score_idx = self.scores.index(max(self.scores))
        return self.models[max_score_idx], self.scores[max_score_idx], self.states[max_score_idx]


def train_test(dataset):
    train_size = int(dataset.shape[0] * 0.8)
    train_sample = np.random.choice(dataset.index, size=train_size)
    train_set = dataset.loc[train_sample, :]
    test_set_sample = dataset.index.difference(train_set.index)
    test_set = dataset.loc[test_set_sample, :]

    train_set = train_set.sort_values(by="time")
    test_set = test_set.sort_values(by="time")
    return train_set, test_set


def extract_column(X_train, X_test, col_name):
    return X_train[[col_name]], X_test[[col_name]]


def select_hidden_state(X_train, X_test):
    models_container = ModelsContainer()
    for hs in HIDDEN_STATES:
        hmm_model = hmm.GaussianHMM(n_components=hs, n_iter=20)
        hmm_model.fit(X_train)
        model_score = hmm_model.score(X_test)
        models_container.add_model(hmm_model, model_score, hs)
    return models_container.get_best_model()


def select_best_model(dataset, col_name, reps=3):
    models_container = ModelsContainer()
    for _ in range(reps):
        X_train, X_test = train_test(dataset)
        X_train, X_test = extract_column(X_train, X_test, col_name)
        model, score, hs = select_hidden_state(X_train, X_test)
        models_container.add_model(model, score, hs)
    return models_container.get_best_model()


def train_device_models(devices_dataset):
    devices_names = devices_dataset.columns[1:]

    collected_models = OrderedDict()
    for devices_cols in devices_dataset.columns[1:]:
        collected_models[devices_cols] = select_best_model(devices_dataset, devices_cols)

    dev_len = len(devices_names)
    results_mtrx = np.zeros((dev_len, dev_len))
    X_train, X_test = train_test(devices_dataset)
    for i, (model_name, model_info) in enumerate(collected_models.items()):
        model = model_info[0]
        for j, dn in enumerate(devices_names):
            _, X_test_col = extract_column(X_train, X_test, dn)
            model_prediction = model.score(X_test_col[[dn]])
            results_mtrx[j, i] = model_prediction

    result_df = pd.DataFrame(results_mtrx)
    result_df.columns = devices_names
    return result_df, collected_models


def summarize_params_selection(collected_models):
    selected_params = [[model_name, model_info[2]] for model_name, model_info in collected_models.items()]
    selected_params = pd.DataFrame(selected_params)
    selected_params.columns = ["Device", "Hidden States"]
    return selected_params.set_index("Device")


def read_test_files(test_dir):
    test_sets_paths = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    test_sets_paths = sorted([os.path.join(test_dir, f) for f in test_sets_paths])
    return [pd.read_csv(f)[["dev"]] for f in test_sets_paths]


def evaluate_models(test_dfs, collected_models, true_labels):

    def evaluate(X):
        score_records = dict()
        for model_name, model_info in collected_models.items():
            model = model_info[0]
            model_score = model.score(X)
            score_records[model_name] = [model_score]
        return pd.DataFrame(score_records)

    evaluation_results = pd.concat([evaluate(X) for X in test_dfs]).round(3)
    evaluation_results["True labels"] = true_labels
    evaluation_results.reset_index(inplace=True, drop=True)
    return evaluation_results
