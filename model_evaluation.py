from aequitas.flow.optimization import Result
from aequitas.flow.methods import PreProcessing, InProcessing, PostProcessing
from aequitas.flow.methods.postprocessing import Threshold
from aequitas.flow.evaluation import evaluate_fairness, evaluate_performance
from datasets import IIDDataset
from sklearn.metrics import confusion_matrix, accuracy_score
from constants import ABBREVIATIONS
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import os


def evaluate_model(
    artifacts_folder: str,
    model: tuple[PreProcessing, InProcessing, PostProcessing, Threshold],
    X: pd.DataFrame,
    y: pd.Series,
    s: pd.Series = None,
    validation=False,
) -> dict:
    """
    Evaluate the performance and fairness of a trained model in a set of data.

    Parameters
    ----------
    artifacts_folder : str
        Folder where the results should be stored.
    model : tuple[PreProcessing, InProcessing, PostProcessing, Threshold]
        Model to be evaluated.
    X : pd.DataFrame
        Features of the data.
    y : pd.Series
        Labels of the data.
    s : pd.Series, optional
        Sensitive attribute of the data, by default None.
    validation : bool, optional
        Whether the data is validation or test data, by default False.

    Returns
    -------
    dict
        Dictionary with the results of the evaluation.
    """
    if model[0].used_in_inference:
        X, y, s = model[0].transform(X.copy(), y.copy(), s.copy())

    y_pred = model[1].predict_proba(X, s)
    y_pred.to_frame().to_csv(
        f"{artifacts_folder}/{'validation' if validation else 'test'}_scores.csv"
    )

    y_pred = model[2].transform(X, y_pred, s)

    if sorted(y_pred.unique().tolist()) != [0, 1]:
        y_pred = model[3].transform(X, y_pred, s)

    y_pred.to_frame().to_csv(
        f"{artifacts_folder}/{'validation' if validation else 'test'}_bin.csv"
    )

    results = evaluate_performance(y, y_pred)

    if s is not None:
        results.update(evaluate_fairness(y, y_pred, s, True))

    return results


def save_noisy_test_results(
    dataset_name: str,
    variant: str,
    labels: list[list[int]],
    noise_rates: list[tuple[int]],
    methods: list[str],
    experiment_name: str,
    n_trials: int,
):
    """
    Take a trained model and evaluate its performance and fairness on noisy validation
    and test sets.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    variant : str
        Variant of the dataset.
    labels : list[list[int]]
        List of cases of Y-dependant noise injection to consider. Each list should
        contain in which classes noise was injected.
    noise_rates : list[tuple[int]]
        List of noise rates to consider.
    methods : list[str]
        List of methods to evaluate.
    experiment_name : str
        Name of the experiment.
    n_trials : int
        Number of trials set for the experiment.

    Examples
    --------
    >>> save_noisy_test_results(
    ...     "BankAccountFraud",
    ...     "TypeII",
    ...     [[0], [1], [0, 1]],
    ...     [(0, 0), (5, 5), (10, 10), (20, 20)],
    ...     ["lightgbm", "OBNC", "Fair-OBNC"],
    ...     "experiment",
    ...     50,
    ... )
    """
    data = IIDDataset(dataset_name, variant)
    data.load_data()
    data.create_splits()

    experiment_folder = f"artifacts/{experiment_name}"

    t = tqdm(total=len(labels) * len(noise_rates) * len(methods) * n_trials)

    for label in labels:
        for noise_rate in noise_rates:
            dataset_folder = (
                f"{ABBREVIATIONS[dataset_name]}_{str.lower(variant)}_"
                f"label_{label[0] if len(label)==1 else 'both'}_"
                f"{noise_rate[0]}_{noise_rate[1]}"
            )

            for method in methods:
                noisy_test_folder = (
                    f"{experiment_folder}_noisy_test/{dataset_folder}/{method}"
                )
                artifacts_folder = f"{experiment_folder}/{dataset_folder}/{method}"

                with open(f"{artifacts_folder}/results.pickle", "rb") as f:
                    clean_results = pickle.load(f)
                noisy_results = []

                for trial_n in range(n_trials):
                    os.makedirs(f"{noisy_test_folder}/{trial_n}", exist_ok=True)

                    with open(
                        f"{artifacts_folder}/{trial_n}/preprocessing.pickle", "rb"
                    ) as f:
                        preprocessing = pickle.load(f)
                    with open(
                        f"{artifacts_folder}/{trial_n}/inprocessing.pickle", "rb"
                    ) as f:
                        inprocessing = pickle.load(f)
                    with open(
                        f"{artifacts_folder}/{trial_n}/postprocessing.pickle", "rb"
                    ) as f:
                        postprocessing = pickle.load(f)
                    with open(
                        f"{artifacts_folder}/{trial_n}/threshold.pickle", "rb"
                    ) as f:
                        threshold = pickle.load(f)

                    model = (preprocessing, inprocessing, postprocessing, threshold)

                    noisy_test_labels = pd.read_csv(
                        f"data/{dataset_name}/{variant}/noisy/label_"
                        f"{label[0] if len(label) == 1 else 'both'}/"
                        f"test_{noise_rate[0]/100}_{noise_rate[1]/100}.csv",
                        index_col=0,
                    )["0"]
                    noisy_val_labels = pd.read_csv(
                        f"data/{dataset_name}/{variant}/noisy/label_"
                        f"{label[0] if len(label) == 1 else 'both'}/validation_"
                        f"{noise_rate[0]/100}_{noise_rate[1]/100}.csv",
                        index_col=0,
                    )["0"]

                    noisy_test_y = data.test.y.copy()
                    noisy_val_y = data.validation.y.copy()

                    if noisy_test_labels.shape[0] > 0:
                        noisy_test_y.loc[noisy_test_labels.index] = (
                            noisy_test_labels.values
                        )

                    if noisy_val_labels.shape[0] > 0:
                        noisy_val_y.loc[noisy_val_labels.index] = (
                            noisy_val_labels.values
                        )

                    test_results = evaluate_model(
                        f"{noisy_test_folder}/{trial_n}",
                        model,
                        data.test.X,
                        noisy_test_y,
                        data.test.s,
                        validation=False,
                    )

                    val_results = evaluate_model(
                        f"{noisy_test_folder}/{trial_n}",
                        model,
                        data.validation.X,
                        noisy_val_y,
                        data.validation.s,
                        validation=True,
                    )

                    noisy_results.append(
                        Result(
                            id=trial_n,
                            test_results=test_results,
                            hyperparameters=clean_results[
                                trial_n
                            ].hyperparameters.copy(),
                            validation_results=val_results,
                        )
                    )
                    t.update()

                with open(f"{noisy_test_folder}/results.pickle", "wb") as f:
                    pickle.dump(noisy_results, f)

    t.close()


def get_transformed_labels(
    dataset_name: str,
    variant: str,
    labels: list[list[int]],
    noise_rates: list[tuple[int]],
    methods: list[str],
    experiment_name: str,
    n_trials: int,
):
    """
    Store the corrected labels obtained by transforming the noisy training set using
    each considered label noise correction method.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    variant : str
        Variant of the dataset.
    labels : list[list[int]]
        List of cases of Y-dependant noise injection to consider. Each list should
        contain in which classes noise was injected.
    noise_rates : list[tuple[int]]
        List of noise rates to consider.
    methods : list[str]
        List of methods to evaluate. Note that only OBNC, Fair-OBNC and Massaging
        transform the labels.
    experiment_name : str
        Name of the experiment.
    n_trials : int
        Number of trials set for the experiment.

    Examples
    --------
    >>> get_transformed_labels(
    ...     "BankAccountFraud",
    ...     "TypeII",
    ...     [[0], [1], [0, 1]],
    ...     [(0, 0), (5, 5), (10, 10), (20, 20)],
    ...     ["OBNC", "Fair-OBNC", "Massaging"],
    ...     "experiment",
    ...     50
    ... )
    """
    data = IIDDataset(dataset_name, variant)
    data.load_data()
    data.create_splits()

    t = tqdm(total=len(labels) * len(noise_rates) * len(methods) * n_trials)

    for label in labels:
        for noise_rate in noise_rates:
            for method in methods:
                artifacts_folder = (
                    f"artifacts/{experiment_name}/{ABBREVIATIONS[dataset_name]}_"
                    f"{str.lower(variant)}_label_"
                    f"{label[0] if len(label)==1 else 'both'}_"
                    f"{noise_rate[0]}_{noise_rate[1]}/{method}"
                )

                for trial_n in range(n_trials):
                    with open(
                        f"{artifacts_folder}/{trial_n}/preprocessing.pickle", "rb"
                    ) as f:
                        preprocessing = pickle.load(f)

                    noisy_labels = pd.read_csv(
                        f"data/{dataset_name}/{variant}/noisy/label_"
                        f'{label[0] if len(label) == 1 else "both"}/train_'
                        f"{noise_rate[0]/100}_{noise_rate[1]/100}.csv",
                        index_col=0,
                    )["0"]
                    noisy_y = data.train.y.copy()
                    if noisy_labels.shape[0] > 0:
                        noisy_y.loc[noisy_labels.index] = noisy_labels.values

                    _, y_transformed, _ = preprocessing.transform(
                        data.train.X, noisy_y, data.train.s
                    )
                    y_transformed.to_csv(
                        f"{artifacts_folder}/{trial_n}/transformed_labels.csv"
                    )

                    t.update()
    t.close()


def fpr(true, corrected):
    cm = confusion_matrix(true, corrected)
    if len(cm) == 2:
        return cm[0][1] / np.sum(cm[0])
    elif len(cm) == 0:
        return 0
    else:
        if corrected.unique()[0] == 0:
            return 0
        else:
            return 1


def fnr(true, corrected):
    cm = confusion_matrix(true, corrected)
    if len(cm) == 2:
        return cm[1][0] / np.sum(cm[1])
    elif len(cm) == 0:
        return 0
    else:
        if corrected.unique()[0] == 0:
            return 1
        else:
            return 0


def save_reconstruction_error_df(
    dataset: str,
    variant: str,
    labels: list[str],
    noise_rates: list[tuple[int]],
    methods: list[str],
    experiment_name: str,
    n_trials: int,
    destination_folder: str = "reconstruction_scores",
):
    """
    Create csv tables with the reconstruction error, FPR, FNR and percentage of flipped
    instances for the conducted experiments.

    Parameters
    ----------
    dataset : str
        Name of the dataset.
    variant : str
        Variant of the dataset.
    labels : list[str]
        List of cases of Y-dependant noise injection to consider. Can include '0', '1'
        or 'both'.
    noise_rates : list[tuple[int]]
        List of noise rates to consider.
    methods : list[str]
        List of methods to evaluate. Note that only OBNC, Fair-OBNC and Massaging
        transform the labels.
    n_trials : int
        Number of trials set for the experiment.
    destination_folder : str, optional
        Folder where the results should be stored, by default "reconstruction_scores".

    Examples
    --------
    >>> save_reconstruction_error_df(
    ...     "BankAccountFraud",
    ...     "TypeII",
    ...     ["0", "1", "both"],
    ...     [(0, 0), (5, 5), (10, 10), (20, 20)],
    ...     ["OBNC", "Fair-OBNC", "Massaging"],
    ...     'noise_injection_experiments',
    ...     50,
    ... )
    """
    data = IIDDataset(dataset, variant)
    data.load_data()
    data.create_splits()

    os.makedirs(destination_folder, exist_ok=True)

    clean_y = data.train.y.copy()

    t = tqdm(total=(len(labels) * len(noise_rates) * len(methods) * n_trials))

    for label in labels:
        os.makedirs(f"{destination_folder}/label_{label}", exist_ok=True)
        for nr in noise_rates:
            df = pd.DataFrame(
                columns=methods,
                index=[
                    "Reconstruction Score",
                    "FPR",
                    "FNR",
                    "Flipped instances",
                    "Flipped instances %",
                    "Reconstruction Score (group 0)",
                    "FPR (group 0)",
                    "FNR (group 0)",
                    "Flipped instances (group 0)",
                    "Flipped instances % (group 0)",
                    "Reconstruction Score (group 1)",
                    "FPR (group 1)",
                    "FNR (group 1)",
                    "Flipped instances (group 1)",
                    "Flipped instances % (group 1)",
                ],
            )

            noisy_labels = pd.read_csv(
                f"data/{dataset}/{variant}/noisy/label_{label}/"
                f"train_{nr[0]/100}_{nr[1]/100}.csv",
                index_col=0,
            )["0"]
            noisy_y = data.train.y.copy()
            if noisy_labels.shape[0] > 0:
                noisy_y.loc[noisy_labels.index] = noisy_labels
            noisy_0 = noisy_y.loc[~data.train.s.astype(bool)]
            noisy_1 = noisy_y.loc[data.train.s]

            metrics = {
                metric: {method: [] for method in methods} for metric in df.index
            }

            for method in methods:
                for trial in range(n_trials):
                    corrected_y = pd.read_csv(
                        f"artifacts/{experiment_name}/"
                        f"{ABBREVIATIONS[dataset]}_{str.lower(variant)}_label_{label}_"
                        f"{nr[0]}_{nr[1]}/{method}/{trial}/transformed_labels.csv",
                        index_col=0,
                    )["fraud_bool"]

                    metrics["Reconstruction Score"][method].append(
                        accuracy_score(clean_y, corrected_y)
                    )
                    metrics["FPR"][method].append(fpr(clean_y, corrected_y))
                    metrics["FNR"][method].append(fnr(clean_y, corrected_y))
                    metrics["Flipped instances"][method].append(
                        (noisy_y != corrected_y).sum()
                    )
                    metrics["Flipped instances %"][method].append(
                        (noisy_y != corrected_y).sum() / noisy_y.shape[0]
                    )

                    clean_0 = clean_y.loc[~data.train.s.astype(bool)]
                    corrected_0 = corrected_y.loc[~data.train.s.astype(bool)]
                    metrics["Reconstruction Score (group 0)"][method].append(
                        accuracy_score(clean_0, corrected_0)
                    )
                    metrics["FPR (group 0)"][method].append(fpr(clean_0, corrected_0))
                    metrics["FNR (group 0)"][method].append(fnr(clean_0, corrected_0))
                    metrics["Flipped instances (group 0)"][method].append(
                        (noisy_0 != corrected_0).sum()
                    )
                    metrics["Flipped instances % (group 0)"][method].append(
                        (noisy_0 != corrected_0).sum() / noisy_0.shape[0]
                    )

                    clean_1 = clean_y.loc[data.train.s]
                    corrected_1 = corrected_y.loc[data.train.s]
                    metrics["Reconstruction Score (group 1)"][method].append(
                        accuracy_score(clean_1, corrected_1)
                    )
                    metrics["FPR (group 1)"][method].append(fpr(clean_1, corrected_1))
                    metrics["FNR (group 1)"][method].append(fnr(clean_1, corrected_1))
                    metrics["Flipped instances (group 1)"][method].append(
                        (noisy_1 != corrected_1).sum()
                    )
                    metrics["Flipped instances % (group 1)"][method].append(
                        (noisy_1 != corrected_1).sum() / noisy_1.shape[0]
                    )

                    t.update()

            for metric, values in metrics.items():
                df.loc[metric] = [np.mean(values[method]) for method in df.columns]
            df.to_csv(
                f"{destination_folder}/label_{label}/noise_rate_{nr[0]}_{nr[1]}.csv",
                index=True,
            )

    t.close()
