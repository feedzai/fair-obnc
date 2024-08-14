from aequitas.flow.experiment import Experiment
from constants import NOISE_RATES
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str)
parser.add_argument("variant", type=str)
parser.add_argument("experiment_name", type=str)
parser.add_argument("--noise-injection", action="store_true")
args = parser.parse_args()

if args.noise_injection:
    for label in ["0", "1", "both"]:
        for noise_rate in NOISE_RATES:
            experiment = Experiment(
                config_file=Path(
                    f"./configs/exp_{args.dataset}_{args.variant}_label_"
                    f"{label}_{noise_rate[0]}_{noise_rate[1]}.yaml"
                ),
                name=args.experiment_name,
            )
            experiment.run()

else:
    experiment = Experiment(
        config_file=Path(f"./configs/exp_{args.dataset}_{args.variant}.yaml"),
        name=args.experiment_name,
    )
    experiment.run()
