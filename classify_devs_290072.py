import argparse
import os
import pandas as pd


from hmmlearn import hmm


def ParseArguments():
    parser = argparse.ArgumentParser(description="Devices classification")
    parser.add_argument('--train', default="./house_devices.csv", required=True)
    parser.add_argument('--test', default="./test_sets", required=True)
    parser.add_argument('--output', default="./results.txt", required=True)
    args = parser.parse_args()
    return args.train, args.test, args.output


def read_test_files(test_dir):
    test_sets_paths = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    test_sets_paths = sorted([os.path.join(test_dir, f) for f in test_sets_paths])
    return [pd.read_csv(f)[["dev"]] for f in test_sets_paths], test_sets_paths


#train_file, test_folder, output_file = ParseArguments()
train_file, test_folder, output_file = ["./house_devices.csv", "./test_sets", "./results.txt"]
test_dfs, file_names = read_test_files(test_folder)
model_params = {"lighting2": 2, "lighting5": 2, "lighting4": 2, "refrigerator": 2, "microwave": 2}


def train_models():
    train_df = pd.read_csv(train_file)
    models_dict = dict()
    for m_name, m_param in model_params.items():
        X_train = train_df[[m_name]]
        hmm_model = hmm.GaussianHMM(n_components=m_param, n_iter=20)
        hmm_model.fit(X_train)
        models_dict[m_name] = hmm_model
    return models_dict


models_dict = train_models()


results_dict = {"file": [], "dev_classified": []}
for test_df, file_name in zip(test_dfs, file_names):
    model_scores = dict()
    X = test_df[["dev"]]
    for model_name, model in models_dict.items():
        model_scores[model_name] = model.score(X)
    max_key = max(model_scores, key=model_scores.get)
    results_dict["file"].append(file_name)
    results_dict["dev_classified"].append(max_key)


results_df = pd.DataFrame.from_dict(results_dict)
with open(output_file, 'w') as f:
    f.write(results_df.to_string(index=False))