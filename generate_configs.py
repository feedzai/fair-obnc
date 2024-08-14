from constants import ABBREVIATIONS, NOISE_RATES
import yaml


def generate_dataset_configs(variants: dict[str, list[str]]):
    """
    Generate the dataset configuration files for the noise injection experiments.

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
    >>> generate_dataset_configs(variants)
    """
    for dataset in variants.keys():
        for variant in variants[dataset]:
            for target_labels in [[0], [1], [0, 1]]:
                for nr in NOISE_RATES:
                    save_dataset_config(dataset, variant, target_labels, nr[0], nr[1])


def save_dataset_config(dataset, variant, target_labels, nr_0, nr_1):
    name = (
        f"{ABBREVIATIONS[dataset]}_{str.lower(variant)}_label_"
        f"{target_labels[0] if len(target_labels) == 1 else 'both'}"
        f"_{nr_0}_{nr_1}"
    )

    config = {
        name: {
            "classpath": "datasets.NoisyDataset",
            "threshold": {
                "threshold_type": "top_pct",
                "threshold_value": 0.01,
            },
            "args": {
                "dataset": dataset,
                "variant": variant,
                "noise_rates": {0: nr_0 / 100, 1: nr_1 / 100},
                "y_dependant": target_labels,
            },
        }
    }

    with open(f"configs/datasets/{name}.yaml", "w") as file:
        yaml.dump(config, file)


def generate_original_dataset_configs(variants: dict[str, list[str]]):
    """
    Generate the dataset configuration files for experiments that do not involve noise
    injection.

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
    >>> generate_original_dataset_configs(variants)
    """
    for dataset in variants.keys():
        for variant in variants[dataset]:
            name = f"{ABBREVIATIONS[dataset]}_{str.lower(variant)}"
            config = {
                name: {
                    "classpath": f"aequitas.flow.datasets.{dataset}",
                    "threshold": {"threshold_type": "top_pct", "threshold_value": 0.01},
                    "args": {"variant": variant, "extension": "csv"},
                }
            }

            with open(f"configs/datasets/{name}.yaml", "w") as file:
                yaml.dump(config, file)


def generate_experiment_files(
    methods: list[str],
    variants: dict[str, list[str]],
    noise_injection: bool,
    n_trials: int,
):
    """
    Generate the experiment configuration files for the noise injection experiments.

    Parameters
    ----------
    methods : list[str]
        The methods to be used in the experiments.
    variants : dict[str, list[str]]
        The variants of the datasets to be used in the experiment.
    noise_injection : bool
        Whether the experiments will involve noise injection or not.
    n_trials : int
        The number of trials to be performed in the experiments.

    Examples
    --------
    >>> methods = ["lightgbm", "OBNC", "Fair-OBNC", "DataRepairer"]
    >>> variants = {
    ...     "BankAccountFraud": ["TypeI", "TypeII", "III"],
    ...     "FolkTables": ["ACSIncome"]
    ... }
    >>> generate_experiment_files(methods, variants, True, 50)
    """
    for dataset in variants.keys():
        for variant in variants[dataset]:
            if noise_injection:
                for labels in ["0", "1", "both"]:
                    for nr in NOISE_RATES:
                        save_experiment_config(
                            methods,
                            dataset_name(dataset, variant, labels, nr[0], nr[1]),
                            n_trials,
                        )
            else:
                save_experiment_config(
                    methods, dataset_name(dataset, variant, "0", 0, 0), n_trials
                )


def dataset_name(dataset, variant, target_labels, nr_0, nr_1):
    return (
        f"{ABBREVIATIONS[dataset]}_{str.lower(variant)}_label_"
        f"{target_labels[0] if len(target_labels) == 1 else 'both'}_{nr_0}_{nr_1}"
    )


def save_experiment_config(methods, dataset, n_trials):
    config = {
        "methods": methods,
        "datasets": [dataset],
        "optimization": {
            "n_trials": n_trials,
            "n_jobs": 1,
            "sampler": "RandomSampler",
            "sampler_args": {"seed": 42},
        },
    }

    with open(f"configs/exp_{dataset}.yaml", "w") as file:
        yaml.dump(config, file)
