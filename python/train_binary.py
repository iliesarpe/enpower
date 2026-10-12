"""Binary experiments: trains the ensembles of one random split and runs the Troupe baselines.

For one OpenML dataset and one seed this script
  1. downloads and preprocesses the dataset (load_and_preprocess_data), binarizing the label
     to "most frequent class" vs "rest";
  2. splits it 75/15/10 into training, D_val and D_test, all drawn with the seed;
  3. trains the requested ensembles (random forest, AdaBoost, heterogeneous) on the training split
     and stores the predictions of every base classifier on D_val and D_test in
     <out-root>/ensembles/<dataset>_<ensemble>_predictions.npz;
  4. runs Tr-PS and Tr-MLE (the library of Rozemberczki et al., in shapley/) on D_val and stores
     their values and runtimes in <out-root>/baselines/binary_hpc_<id>[_<ensemble>].json.

Every other method of the binary experiments (ENPOWER, WMG, ISCG, MC) runs in the C++ library on
the stored predictions, through `cpp/tools/enpower.py splits run|mc`.

    python train_binary.py --dataset_id 31 --seed 100 --out-root ../outputs/splits/seed100
"""
import numpy as np
import os
import sys
import time
import json
import warnings
import argparse

# scikit-learn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ensembles and base classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# The Troupe library (Rozemberczki et al.), bundled unmodified in shapley/
current_dir = os.path.dirname(os.path.abspath(__file__))
troupe_repo_root = os.path.join(current_dir, 'shapley')
if troupe_repo_root not in sys.path:
    sys.path.insert(0, troupe_repo_root)
from shapley import PermutationSampler, MultilinearExtension

warnings.filterwarnings('ignore')

# --- configuration ---
REPO = os.path.dirname(current_dir)
RSEED = 100
PS_SAMPLES = 1000          # permutations of Tr-PS
TROUPE_QUOTA = 0.5 + 1e-5  # strict majority of the n base classifiers

# The datasets of the paper, by OpenML id
ID_TO_NAME = {
    31: 'credit-g',
    1461: 'bank-marketing',
    1464: 'blood-transfusion-service-center',
    1468: 'cnae-9',
    40945: 'titanic',
    150: 'Covertype',
}


def get_heterogeneous_ensemble(seed=RSEED):
    models = []
    for i in range(10): models.append((f'dt_{i}', DecisionTreeClassifier(max_depth=i+3, random_state=seed+i)))
    for i in range(5): models.append((f'svm_sgd_{i}', SGDClassifier(loss='hinge', alpha=0.0001*(10**i), max_iter=200, random_state=seed+i)))
    for i in range(5): models.append((f'lr_sgd_{i}', SGDClassifier(loss='log_loss', alpha=0.0001*(10**i), max_iter=200, random_state=seed+i)))
    for i in range(5): models.append((f'hgb_{i}', HistGradientBoostingClassifier(max_iter=int(50 + (i*1.2)), random_state=seed+i)))
    for i in range(5): models.append((f'mlp_{i}', MLPClassifier(hidden_layer_sizes=(16,), max_iter=100, random_state=seed+i)))
    return VotingClassifier(estimators=models, voting='hard')


def load_and_preprocess_data(data_id, seed=RSEED):
    """The dataset of an OpenML id, preprocessed, binarized and split 75/25 by the seed.
    The 25% is split again into D_val and D_test (60/40) after training."""
    print(f"Fetching OpenML Dataset ID: {data_id}...")
    dataset = fetch_openml(data_id=data_id, as_frame=True, parser='auto')

    X = dataset.data
    y = dataset.target
    mode_class = y.mode()[0]
    y_binary = (y == mode_class).astype(int).values

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    X_processed = preprocessor.fit_transform(X)
    return train_test_split(X_processed, y_binary, test_size=0.25, random_state=seed), dataset.details['name']


def extract_predictions(estimators, X):
    preds = [est.predict(X) for est in estimators]
    return np.column_stack(preds)


def main():
    parser = argparse.ArgumentParser(description="Train the ensembles of one split (binary labels) and run Troupe")
    parser.add_argument('--dataset_id', type=int, required=True, help="OpenML id of the dataset")
    # The seed determines the split and seeds every base classifier; --out-root
    # separates the outputs of the splits, e.g., outputs/splits/seed101.
    parser.add_argument('--seed', type=int, default=RSEED)
    parser.add_argument('--out-root', default=os.path.join(REPO, "outputs", "splits", "seed%d" % RSEED))
    parser.add_argument('--ensembles', nargs='+', choices=['random_forest', 'adaboost', 'heterogeneous'],
                        default=['random_forest', 'adaboost', 'heterogeneous'])
    parser.add_argument('--ps-samples', type=int, default=PS_SAMPLES, help="permutations of Tr-PS")
    parser.add_argument('--mle-runs', type=int, default=1, help="runs of Tr-MLE (it is deterministic)")
    args = parser.parse_args()
    seed = args.seed

    EXPPATH = os.path.join(args.out_root, "ensembles")
    EXPRES_FOLDER = os.path.join(args.out_root, "baselines")
    RESULTS_FILENAME = f"binary_hpc_{args.dataset_id}"
    # one file per ensemble when the ensembles run as separate jobs, so that
    # concurrent jobs never write the same file
    if len(args.ensembles) == 1:
        RESULTS_FILENAME += f"_{args.ensembles[0]}"
    RESULTS_FILE = os.path.join(EXPRES_FOLDER, RESULTS_FILENAME + ".json")

    os.makedirs(EXPPATH, exist_ok=True)
    os.makedirs(EXPRES_FOLDER, exist_ok=True)

    all_results = {}
    data_id = args.dataset_id
    d_name = ID_TO_NAME.get(data_id, f"dataset_{data_id}")

    ensembles = {
        "random_forest": RandomForestClassifier(n_estimators=50, random_state=seed),
        "adaboost": AdaBoostClassifier(n_estimators=150, random_state=seed),
        "heterogeneous": get_heterogeneous_ensemble(seed)
    }
    ensembles = {k: v for k, v in ensembles.items() if k in args.ensembles}

    # the dataset is loaded only if the predictions of an ensemble are not stored yet
    need_raw_data = any(not os.path.exists(os.path.join(EXPPATH, f"{d_name}_{e}_predictions.npz"))
                        for e in ensembles)
    if need_raw_data:
        (X_train, X_test, y_train, y_test), _ = load_and_preprocess_data(data_id, seed)
        test_samples_log = len(y_test)
    else:
        print("  > Predictions of every ensemble already stored, the dataset is not needed.")
        test_samples_log = "stored"

    print(f"\n{'#'*60}\nProcessing Dataset: {d_name.upper()} | D_val + D_test: {test_samples_log}\n{'#'*60}")
    all_results[d_name] = {"ensembles": {}}

    for ens_name, model in ensembles.items():
        print(f"\n--- Training/Loading {ens_name.upper()} ---")
        npz_path = os.path.join(EXPPATH, f"{d_name}_{ens_name}_predictions.npz")

        if os.path.exists(npz_path):
            print(f"  > Loading the stored predictions of {ens_name.upper()}...")
            with np.load(npz_path) as data:
                preds_val = data['preds_val']
                y_val = data['labels_val']
                preds_test = data['preds_test']
                y_test_final = data['labels_test']
        else:
            print(f"  > Training {ens_name.upper()}...")
            model.fit(X_train, y_train)

            # predictions on the held-out 25%
            preds_25 = extract_predictions(model.estimators_, X_test)

            # split the held-out 25% into D_val (15%) and D_test (10%)
            preds_val, preds_test, y_val, y_test_final = train_test_split(
                preds_25, y_test, test_size=0.40, random_state=seed
            )

            np.savez_compressed(npz_path,
                                preds_val=preds_val, labels_val=y_val,
                                preds_test=preds_test, labels_test=y_test_final)

        M = preds_val.shape[1]

        acc = accuracy_score(y_test_final, (np.sum(preds_test, axis=1) > (M // 2)).astype(int))
        print(f"  > M={M}. Test accuracy={acc:.4f}. |D_val|={len(y_val)}")

        ens_results = {"M": M, "accuracy": acc, "seed": seed, "solvers": {}}

        # Troupe: weight 1/M for a correct classifier on a data point, and a strict majority quota
        print("  > Running Troupe PermutationSampler...")
        W_matrix = np.where(preds_val == y_val[:, None], 1.0/M, 0.0)
        troupe_solver = PermutationSampler(args.ps_samples, seed)
        start_t = time.time()
        troupe_solver.solve_game(W_matrix, q=TROUPE_QUOTA)
        troupe_vals = troupe_solver.get_solution().mean(axis=0)
        ens_results["solvers"]["Troupe_Permutation"] = {"vals": troupe_vals.tolist(), "time": time.time() - start_t,
                                                        "permutations": args.ps_samples}

        print(f"  > Running MultilinearExtension ({args.mle_runs} runs)...")
        mle_times = []
        mle_results = []
        for i in range(args.mle_runs):
            try:
                mle_solver = MultilinearExtension()
                start_t = time.time()
                mle_solver.solve_game(W_matrix, q=TROUPE_QUOTA)
                mle_times.append(time.time() - start_t)
                mle_results.append(mle_solver.get_solution().mean(axis=0))
            except Exception as e:
                print(f"    -> MLE Run {i} failed: {e}")

        if mle_results:
            ens_results["solvers"]["Troupe_MLE"] = {
                "vals": np.median(mle_results, axis=0).tolist(),
                "time_sec": np.mean(mle_times)
            }

        all_results[d_name]["ensembles"][ens_name] = ens_results

        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=4)

    print(f"\n{'='*50}\nDone. Results in {RESULTS_FILE}")


if __name__ == "__main__":
    main()
