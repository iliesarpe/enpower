"""Multiclass experiments: trains the ensembles of one random split on the original labels.

For one OpenML dataset and one seed this script downloads and preprocesses the dataset
(load_and_preprocess_data, the labels kept as 0, ..., L-1), splits it 75/15/10 into training,
D_val and D_test with the seed, trains the requested ensembles on the training split, and stores
the predictions of every base classifier on D_val and D_test in
<out-root>/ensembles_multiclass/<dataset>_<ensemble>_predictions.npz.

The multiclass indices are then computed and timed by the C++ library on the stored predictions,
through `cpp/tools/enpower.py mc run`.

    python train_multiclass.py --dataset_id 1468 --seed 100 --out-root ../outputs/mc_splits/seed100
"""
import numpy as np
import os
import warnings
import argparse

# scikit-learn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ensembles and base classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings('ignore')

# --- configuration ---
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RSEED = 100

ID_TO_NAME = {
    31: 'credit-g',
    1461: 'bank-marketing',
    1464: 'blood-transfusion-service-center',
    1468: 'cnae-9',
    40945: 'titanic',
    150: 'Covertype'
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
    """The dataset of an OpenML id, preprocessed, with its labels encoded as 0, ..., L-1, and
    split 75/25 by the seed. The 25% is split again into D_val and D_test (60/40) after training."""
    print(f"Fetching OpenML Dataset ID: {data_id}...")
    dataset = fetch_openml(data_id=data_id, as_frame=True, parser='auto')

    X = dataset.data
    y = dataset.target

    # encode the classes as integers 0, 1, ..., L-1
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

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
    return train_test_split(X_processed, y_encoded, test_size=0.25, random_state=seed), dataset.details['name']


def extract_predictions(estimators, X):
    preds = [est.predict(X) for est in estimators]
    return np.column_stack(preds)


def main():
    parser = argparse.ArgumentParser(description="Train the ensembles of one split (multiclass labels)")
    parser.add_argument('--dataset_id', type=int, required=True, help="OpenML id of the dataset")
    # The seed determines the split and seeds every base classifier; --out-root
    # separates the outputs of the splits, e.g., outputs/mc_splits/seed101.
    parser.add_argument('--seed', type=int, default=RSEED)
    parser.add_argument('--out-root', default=os.path.join(REPO, "outputs", "mc_splits", "seed%d" % RSEED))
    parser.add_argument('--ensembles', nargs='+', choices=['random_forest', 'adaboost', 'heterogeneous'],
                        default=['random_forest', 'adaboost', 'heterogeneous'])
    args = parser.parse_args()
    seed = args.seed

    data_id = args.dataset_id
    d_name = ID_TO_NAME.get(data_id, f"dataset_{data_id}")

    EXPPATH = os.path.join(args.out_root, "ensembles_multiclass")
    os.makedirs(EXPPATH, exist_ok=True)

    ensembles = {
        "random_forest": RandomForestClassifier(n_estimators=50, random_state=seed),
        "adaboost": AdaBoostClassifier(n_estimators=150, random_state=seed),
        "heterogeneous": get_heterogeneous_ensemble(seed)
    }
    ensembles = {k: v for k, v in ensembles.items() if k in args.ensembles
                 and not os.path.exists(os.path.join(EXPPATH, f"{d_name}_{k}_predictions.npz"))}
    if not ensembles:
        print("Predictions of every ensemble already stored.")
        return

    (X_train, X_test, y_train, y_test), _ = load_and_preprocess_data(data_id, seed)
    print(f"\n{'#'*60}\nProcessing Dataset: {d_name.upper()} | D_val + D_test: {len(y_test)}\n{'#'*60}")

    for ens_name, model in ensembles.items():
        print(f"\n--- Training {ens_name.upper()} ---")
        npz_path = os.path.join(EXPPATH, f"{d_name}_{ens_name}_predictions.npz")
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

        L = int(max(np.max(preds_val), np.max(y_val)) + 1)
        print(f"  > M={preds_val.shape[1]}, L={L}, |D_val|={len(y_val)} -> {npz_path}")


if __name__ == "__main__":
    main()
