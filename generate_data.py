from aequitas.flow.datasets.dataset import Dataset
from aequitas.flow.datasets import BankAccountFraud
from constants import CLASSES, NOISE_RATES
from pathlib import Path
import pandas as pd
import math
import random
import os
import numpy as np


def generate_data(variants: dict[str, list[str]]):
    """
    Store the IID and noisy versions of the datasets in the data folder.

    Parameters
    ----------
    variants : dict[str, list[str]]
        The variants of the datasets to be used in the experiment.

    Examples
    --------
    >>> variants = {
    ...     "BankAccountFraud": ["TypeI", "TypeII", "III"],
    ...     "FolkTables": ["ACSIncome"]
    ... }
    >>> generate_data(variants)
    """
    for dataset_name in variants.keys():
        for variant in variants[dataset_name]:
            dataset = CLASSES[dataset_name](
                variant=variant,
                path=Path(f"datasets/{dataset_name}"),
                extension="parquet",
            )
            dataset.load_data()
            dataset.create_splits()

            train, validation, test = make_dataset_iid(dataset)

            train = train.reset_index(drop=True)
            validation = validation.reset_index(drop=True)
            validation.set_index(validation.index + train.index[-1] + 1, inplace=True)
            test = test.reset_index(drop=True)
            test.set_index(test.index + validation.index[-1] + 1, inplace=True)

            splits = {"train": train, "validation": validation, "test": test}

            os.makedirs(f"data/{dataset_name}/{variant}/iid/", exist_ok=True)

            for split, data in splits.items():
                data.to_csv(
                    f"data/{dataset_name}/{variant}/iid/{split}.csv", index=False
                )

                for y_dependant in [[0], [1], [0, 1]]:
                    for noise_rate in NOISE_RATES:
                        noisy_labels = noise_injection(
                            data[dataset.label_column],
                            data[dataset.sensitive_column],
                            noise_rate,
                            y_dependant,
                        )

                        path = f"data/{dataset_name}/{variant}/noisy/label_{y_dependant[0] if len(y_dependant) == 1 else 'both'}"
                        os.makedirs(path, exist_ok=True)

                        noisy_labels.to_csv(
                            path
                            + f"/{split}_{noise_rate[0]/100}_{noise_rate[1]/100}.csv",
                            index=True,
                        )


def make_dataset_iid(
    dataset: Dataset,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate the IID version of the dataset.

    Parameters
    ----------
    dataset : Dataset
        The Aequitas dataset to be made IID.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The IID version of the dataset, split into train, validation, and test.
    """
    new_data = dataset.data.copy()

    # shuffle the sensitive attribute
    new_data[dataset.sensitive_column] = np.random.permutation(
        new_data[dataset.sensitive_column].values
    )

    if isinstance(dataset, BankAccountFraud):
        new_data = new_data.drop(columns=["month"])
        if dataset.sensitive_column == "customer_age":
            new_data["customer_age_bin"] = new_data["customer_age"].apply(
                lambda x: 1 if x >= dataset.age_cutoff else 0
            )

    # shuffle the splits
    new_data["split"] = np.random.permutation(
        [
            (
                "train"
                if i in dataset.train.index
                else "test" if i in dataset.test.index else "val"
            )
            for i in dataset.data.index
        ]
    )

    train = new_data[new_data["split"] == "train"].drop(columns="split")
    validation = new_data[new_data["split"] == "val"].drop(columns="split")
    test = new_data[new_data["split"] == "test"].drop(columns="split")

    return train, validation, test


def noise_injection(
    y: pd.Series,
    s: pd.Series,
    noise_rate: tuple[int, int],
    y_dependant: list[int],
) -> pd.Series:
    """
    Inject label noise at the specified rate to each sensitive group and to each
    specified class.

    Parameters
    ----------
    y : pd.Series
        The labels.
    s : pd.Series
        The sensitive attribute.
    noise_rate : dict
        The noise rates to be applied to each of the sensitive groups.
        Ex: {0: 0.05, 1: 0.1}
    y_dependant : list[int]
        The classes to be injected with noise. Should be either [0], [1] or [0, 1].

    Returns
    -------
    pd.Series
        The noisy labels.
    """
    noisy_idxs = []
    noisy_labels = []

    for group in range(len(noise_rate)):
        for label in y_dependant:
            y_group = y[(s == group) & (y == label)]
            idxs, labels = random_noise_injection(y_group, noise_rate[group]/100)

            noisy_idxs += idxs
            noisy_labels += labels

    return pd.Series(noisy_labels, index=noisy_idxs)


def random_noise_injection(y: pd.Series, noise_rate: float) -> pd.Series:
    k = math.ceil(y.shape[0] * noise_rate)
    noisy_idxs = random.sample(list(y.index), k)
    noisy_labels = list(1 - y.loc[noisy_idxs].values)

    return noisy_idxs, noisy_labels
