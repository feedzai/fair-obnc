from aequitas.flow.plots.pareto import Plot
from aequitas.flow.utils import read_results
from aequitas.flow.evaluation import Result
from typing import Literal, Optional
from constants import NOISE_TYPES, MARKERS, COLORS, METRICS
import matplotlib.pyplot as plt
import numpy as np
import statistics
import math


def define_limits(
    y_lim_max: tuple[float, float], y_lim_min: tuple[float, float]
) -> tuple[float, float]:
    """
    Calculate the adjusted limits for the y axis of a plot so that both subplots have
    the same range.

    Parameters
    ----------
    y_lim_max : tuple[float, float]
        The limits of the subplot with the widest range.
    y_lim_min : tuple[float, float]
        The limits of the subplot with the narrowest range.

    Returns
    -------
    tuple[float, float]
        The limits for the shared y axis.
    """
    max_range = y_lim_max[1] - y_lim_max[0]
    min_range = y_lim_min[1] - y_lim_min[0]

    range_dif = max_range - min_range

    if y_lim_min[0] - range_dif / 2 < 0:
        return (0, max_range)
    elif y_lim_min[1] + range_dif / 2 > 1:
        return (1 - max_range, 1)
    else:
        return (y_lim_min[0] - range_dif / 2, y_lim_min[1] + range_dif / 2)


def plot_results_single_metric(
    results: dict[str, dict[str, Result]],
    dataset: str,
    variant: str,
    noise_type: Literal["I", "II", "III"],
    label: Literal["0", "1", "both"],
    metric: Literal[
        "Demographic Parity",
        "Equal Opportunity",
        "Predictive Equality",
        "TPR",
        "Accuracy",
        "FPR",
        "FNR",
        "Precision",
    ],
    methods: Optional[list[str]] = None,
    y_lim: Optional[list[int]] = None,
    save_file: Optional[str] = None,
):
    """
    Plot the results according to the specified metric over the chosen noise rates.

    Parameters
    ----------
    results : dict[str, dict[str, Result]]
        The results of the experiment.
    dataset : str
        The abbreviated and lowercase name of the dataset used in the experiment to
        visualize. Ex: "baf".
    variant : str
        The lowercase name of the variant of the dataset used in the experiment to
        visualize. Ex: "typeii".
    noise_type : Literal["I", "II", "III"]
        The type of noise injected in the dataset. The noise rates of each type are
        defined in the constants file.
    label : Literal["0", "1", "both"]
        The class in which the noise was injected.
    metric : Literal["Demographic Parity","Equal Opportunity","Predictive Equality",
                    "TPR","Accuracy","FPR","FNR","Precision"]
        The metric to visualize.
    methods : Optional[list[str]], optional
        The methods to visualize. If None, all methods are visualized, by default None.
    y_lim : Optional[list[int]], optional
        The limits for the y axis. If None, the limits are automatically defined, by
        default None.
    """
    name = f"{dataset}_{variant}_label_{label}"

    if methods is None:
        methods = [
            "lightgbm",
            "OBNC",
            "Fair-OBNC",
            "PrevalenceSampling",
            "Massaging",
            "DataRepairer",
            "CorrelationSuppression",
            "FeatureImportanceSuppression",
        ]

    for method in methods:
        metric_results = []

        for nr in NOISE_TYPES[noise_type]:
            metric_avg = []
            for result in results[f"{name}_{nr[0]}_{nr[1]}"][method]:
                metric_avg.append(result.test_results[METRICS[metric]])

            mean = statistics.mean(metric_avg)
            stdev = statistics.stdev(metric_avg)
            confidence_interval = 1.96 * stdev / math.sqrt(len(metric_avg))
            top = mean - confidence_interval
            bottom = mean + confidence_interval
            plt.plot([f"{nr[1]}%", f"{nr[1]}%"], [top, bottom], color=COLORS[method])

            metric_results.append(np.mean(metric_avg))

        x_labels = [f"{nr[1]}%" for nr in NOISE_TYPES[noise_type]]
        plt.plot(
            x_labels,
            metric_results,
            color=COLORS[method],
            label=("No preprocessing" if method == "lightgbm" else method),
        )

    if y_lim is not None:
        plt.ylim(y_lim)

    plt.xlabel("Noise rate")
    plt.ylabel(metric)
    plt.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, -0.4))
    if save_file:
        plt.savefig(f"{save_file}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_results(
    results: dict[str, dict[str, Result]],
    dataset: str,
    variant: str,
    noise_type: Literal["I", "II", "III"],
    label: Literal["0", "1", "both"],
    fairness_metric: Literal[
        "Demographic Parity", "Equal Opportunity", "Predictive Equality"
    ] = "Demographic Parity",
    performance_metric: Literal["TPR", "Accuracy", "FPR", "FNR", "Precision"] = "TPR",
    methods: list[str] = None,
):
    """
    Plot the results of the experiment according to the specified fairness metric and
    performance metric over the chosen noise rates.

    Parameters
    ----------
    results : dict[str, dict[str, Result]]
        The results of the experiment.
    dataset : str
        The abbreviated and lowercase name of the dataset used in the experiment to
        visualize. Ex: "baf".
    variant : str
        The lowercase name of the variant of the dataset used in the experiment to
        visualize. Ex: "typeii".
    noise_type : Literal["I", "II", "III"]
        The type of noise injected in the dataset. The noise rates of each type are
        defined in the constants file.
    label : Literal["0", "1", "both"]
        The class in which the noise was injected.
    fairness_metric : Literal["Demographic Parity", "Equal Opportunity",
                            "Predictive Equality"]
        The fairness metric to visualize.
    performance_metric : Literal["TPR", "Accuracy", "FPR", "FNR", "Precision"]
        The performance metric to visualize.
    methods : list[str], optional
        The methods to visualize. If None, all methods are visualized, by default None.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    name = f"{dataset}_{variant}_label_{label}"

    if methods is None:
        methods = list(results[list(results.keys())[0]].keys())

    for method in methods:
        fairness = []
        performance = []

        for nr in NOISE_TYPES[noise_type]:
            chosen = Plot(
                results,
                f"{name}_{nr[0]}_{nr[1]}",
                fairness_metric,
                performance_metric,
                method=method,
                split="test",
                alpha=1,
            ).best_model_details
            fairness.append(chosen[fairness_metric])
            performance.append(chosen[performance_metric])

        x_labels = [f"({nr[0]}%,{nr[1]}%)" for nr in NOISE_TYPES[noise_type]]
        axs[0].plot(x_labels, fairness, color=COLORS[method])
        axs[1].plot(
            x_labels,
            performance,
            label=f"{method}",
            color=COLORS[method],
        )

    ax0_range = axs[0].get_ylim()[1] - axs[0].get_ylim()[0]
    ax1_range = axs[1].get_ylim()[1] - axs[1].get_ylim()[0]
    if ax0_range > ax1_range:
        axs[1].set_ylim(define_limits(axs[0].get_ylim(), axs[1].get_ylim()))
    elif ax1_range > ax0_range:
        axs[0].set_ylim(define_limits(axs[1].get_ylim(), axs[0].get_ylim()))

    axs[0].set_xlabel("Noise rate (group 0, group 1)")
    axs[0].set_ylabel(fairness_metric)
    axs[0].set_title("Fairness")

    axs[1].set_xlabel("Noise rate (group 0, group 1)")
    axs[1].set_ylabel(performance_metric)
    axs[1].set_title("Predictive Performance")
    axs[1].legend()

    label_dependant_text = "both classes" if label == "both" else f"class {label}"
    plt.suptitle(
        f"Dataset {dataset} ({variant}) - Noise type {noise_type} - "
        f"Noise in {label_dependant_text}"
    )
    plt.show()


def plot_confidence_interval(
    x: list[float], y: list[float], ax: plt.Axes, color: str, z: float = 1.96
):
    """
    Plot the confidence interval around the mean of the given data.

    Parameters
    ----------
    x : list[float]
        The x axis data.
    y : list[float]
        The y axis data.
    ax : plt.Axes
        The axis to plot the confidence interval.
    color : str
        The color to be used in plotting.
    z : float, optional
        The z value for the confidence interval, by default 1.96.
    """
    mean_x = statistics.mean(x)
    stdev_x = statistics.stdev(x)
    confidence_interval_x = z * stdev_x / math.sqrt(len(x))

    mean_y = statistics.mean(y)
    stdev_y = statistics.stdev(y)
    confidence_interval_y = z * stdev_y / math.sqrt(len(y))

    top_x = mean_x - confidence_interval_x
    bottom_x = mean_x + confidence_interval_x

    top_y = mean_y - confidence_interval_y
    bottom_y = mean_y + confidence_interval_y

    ax.plot([mean_x, mean_x], [top_y, bottom_y], color=color)
    ax.plot([top_x, bottom_x], [mean_y, mean_y], color=color)


def plot_separate_noise_rates(
    results: dict[str, dict[str, Result]],
    dataset: str,
    variant: str,
    noise_type: Literal["I", "II", "III"],
    label: Literal["0", "1", "both"],
    fairness_metric: Literal[
        "Demographic Parity", "Equal Opportunity", "Predictive Equality"
    ] = "Demographic Parity",
    performance_metric: Literal["TPR", "Accuracy", "FPR", "FNR", "Precision"] = "TPR",
    methods=None,
):
    """
    Plot the results of the experiment for each noise rate on a separate subplot.

    Parameters
    ----------
    results : dict[str, dict[str, Result]]
        The results of the experiment.
    dataset : str
        The abbreviated and lowercase name of the dataset used in the experiment to
        visualize. Ex: "baf".
    variant : str
        The lowercase name of the variant of the dataset used in the experiment to
        visualize. Ex: "typeii".
    noise_type : Literal["I", "II", "III"]
        The type of noise injected in the dataset. The noise rates of each type are
        defined in the constants file.
    label : Literal["0", "1", "both"]
        The class in which the noise was injected.
    fairness_metric : Literal["Demographic Parity", "Equal Opportunity",
                            "Predictive Equality"]
        The fairness metric to visualize.
    performance_metric : Literal["TPR", "Accuracy", "FPR", "FNR", "Precision"]
        The performance metric to visualize.
    methods : list[str], optional
        The methods to visualize. If None, all methods are visualized, by default None.
    """
    fig, axs = plt.subplots(1, 4, figsize=(25, 5), sharex=True, sharey=True)
    fig.tight_layout()

    name = f"{dataset}_{variant}_label_{label}"

    if methods is None:
        methods = list(results[list(results.keys())[0]].keys())

    for i in range(4):
        nr = NOISE_TYPES[noise_type][i]
        for method in methods:
            avg_fairness = []
            avg_performance = []

            for result in results[f"{name}_{nr[0]}_{nr[1]}"][method]:
                avg_fairness.append(result.test_results["pprev_ratio"])
                avg_performance.append(result.test_results["tpr"])

            plot_confidence_interval(
                avg_performance, avg_fairness, axs[i], COLORS[method]
            )
            axs[i].plot(
                np.mean(avg_performance),
                np.mean(avg_fairness),
                color=COLORS[method],
                marker="o",
                markersize=10,
            )

        axs[i].set_xlabel(performance_metric)
        axs[i].set_title(f"Noise rate:{nr}")

    axs[0].set_ylabel(fairness_metric)

    label_dependant_text = "both classes" if label == "both" else f"class {label}"
    plt.suptitle(
        f"Dataset {dataset} ({variant}) - Noise type {noise_type} - "
        f"Noise in {label_dependant_text}",
        y=1.1,
    )
    plt.show()


def plot_fairness_performance_tradeoffs(
    results_folder: str,
    dataset: str,
    variant: str,
    label: Literal["0", "1", "both"],
    noise_type: Literal["I", "II", "III"],
    methods: list[str] = None,
    fairness_metric: Literal[
        "Demographic Parity", "Equal Opportunity", "Predictive Equality"
    ] = "Demographic Parity",
    performance_metric: Literal["TPR", "Accuracy", "FPR", "FNR", "Precision"] = "TPR",
    top_k_models: Optional[int] = None,
    noisy_validation: bool = True,
    legend_cols=3,
    legend_pos=(-0.2, -0.4),
    save_file: Optional[str] = None,
):
    """
    Plot the tradeoff between fairness and performance for the chosen methods and noise
    rates.

    Parameters
    ----------
    results_folder : str
        The folder where the results are stored. Ex: "artifacts/experiment".
    dataset : str
        The abbreviated and lowercase name of the dataset used in the experiment to
        visualize. Ex: "baf".
    variant : str
        The lowercase name of the variant of the dataset used in the experiment to
        visualize. Ex: "typeii".
    label : Literal["0", "1", "both"]
        The class in which the noise was injected.
    noise_type : Literal["I", "II", "III"]
        The type of noise injected in the dataset. The noise rates of each type are
        defined in the constants file.
    methods : list[str], optional
        The methods to visualize. If None, all methods are visualized, by default None.
    fairness_metric : Literal["Demographic Parity", "Equal Opportunity",
                            "Predictive Equality"]
        The fairness metric to visualize.
    performance_metric : Literal["TPR", "Accuracy", "FPR", "FNR", "Precision"]
        The performance metric to visualize.
    top_k_models : Optional[int], optional
        The number of models to choose according to their performance on a validation
        set. If None, all models are considered, by default None.
    noisy_validation : bool, optional
        If True, the models are chosen from the noisy validation set. If False, the
        models are chosen from the clean validation set, by default True.
    """
    name = f"{dataset}_{variant}_label_{label}"

    results = read_results(results_folder)
    if top_k_models:
        noisy_results = read_results(f"{results_folder}_noisy_test")

    if methods is None:
        methods = list(results[list(results.keys())[0]].keys())

    noise_rate_legend = False
    for method in methods:
        fairness = []
        performance = []
        for i in range(4):
            nr = NOISE_TYPES[noise_type][i]
            avg_fairness = []
            avg_performance = []

            if top_k_models is None:
                chosen_models = results[f"{name}_{nr[0]}_{nr[1]}"][method]
            else:
                if noisy_validation:
                    all_models = {
                        result.id: result
                        for result in noisy_results[f"{name}_{nr[0]}_{nr[1]}"][method]
                    }
                else:
                    all_models = {
                        result.id: result
                        for result in results[f"{name}_{nr[0]}_{nr[1]}"][method]
                    }
                chosen_models = [
                    v
                    for k, v in sorted(
                        all_models.items(),
                        key=lambda item: item[1].validation_results[
                            METRICS[performance_metric]
                        ],
                        reverse=True,
                    )
                ][:top_k_models]

            for result in chosen_models:
                avg_fairness.append(result.test_results[METRICS[fairness_metric]])
                avg_performance.append(result.test_results[METRICS[performance_metric]])

            fairness.append(np.mean(avg_fairness))
            performance.append(np.mean(avg_performance))
            if len(chosen_models) > 1:
                plot_confidence_interval(
                    avg_performance, avg_fairness, plt.gca(), COLORS[method]
                )
            if not noise_rate_legend:
                plt.plot(
                    np.mean(avg_performance),
                    np.mean(avg_fairness),
                    marker=MARKERS[i],
                    markersize=5,
                    color="black",
                    label=f"Noise rate: {nr[1]}%",
                )
            plt.plot(
                np.mean(avg_performance),
                np.mean(avg_fairness),
                marker=MARKERS[i],
                markersize=7,
                color=COLORS[method],
            )
        noise_rate_legend = True

        plt.plot(
            performance,
            fairness,
            color=COLORS[method],
            label=("No preprocessing" if method == "lightgbm" else method),
        )
    plt.xlabel(performance_metric)
    plt.ylabel(fairness_metric)

    # label_dependant_text = "both classes" if label == "both" else f"class {label}"
    # plt.title(f"Noise in {label_dependant_text}")
    plt.legend(loc="lower left", ncol=legend_cols, bbox_to_anchor=legend_pos)
    if save_file:
        plt.savefig(f"{save_file}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
