from aequitas.flow.datasets import GenericDataset
from constants import (
    VARIANTS,
    SENSITIVE_COLUMN,
    LABEL_COLUMN,
    BOOL_COLUMNS,
    CATEGORICAL_COLUMNS,
)
from typing import Literal, Optional
import pandas as pd


class IIDDataset(GenericDataset):
    def __init__(
        self,
        dataset: Literal["BankAccountFraud", "FolkTables"],
        variant: str,
        label_column: Optional[str] = None,
        sensitive_column: Optional[str] = None,
    ):
        """
        Generate an IID version of the desired Aequitas Dataset.

        Parameters
        ----------
        dataset : Literal["BankAccountFraud", "FolkTables"]
            The Aequitas Dataset.
        variant : str
            The variant of the dataset.
        label_column : Optional[str], optional
            The label column of the dataset. By default, None, which will use the
            dataset's default label column.
        sensitive_column : Optional[str], optional
            The sensitive column of the dataset. By default, None, which will use the
            dataset's default sensitive column.
        """
        if variant not in VARIANTS[dataset]:
            raise ValueError(
                f"For the {dataset} dataset, variant must be one of {VARIANTS[dataset]}"
            )

        if label_column is None:
            if dataset == "BankAccountFraud":
                label_column = LABEL_COLUMN[dataset]
            else:
                label_column = LABEL_COLUMN[dataset][variant]

        if sensitive_column is None:
            sensitive_column = SENSITIVE_COLUMN[dataset]

        train_path = f"data/{dataset}/{variant}/iid/train.csv"
        validation_path = f"data/{dataset}/{variant}/iid/validation.csv"
        test_path = f"data/{dataset}/{variant}/iid/test.csv"

        super().__init__(
            label_column=label_column,
            sensitive_column=sensitive_column,
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            extension="csv",
        )

        self.dataset = dataset
        self.variant = variant

    def load_data(self) -> None:
        """Load the defined dataset."""
        super().load_data()

        if self.dataset == "BankAccountFraud":
            self.data[CATEGORICAL_COLUMNS[self.dataset]] = self.data[
                CATEGORICAL_COLUMNS[self.dataset]
            ].astype("category")
            self.data["customer_age_bin"] = self.data["customer_age_bin"].astype(
                "category"
            )
        else:
            self.data[CATEGORICAL_COLUMNS[self.dataset][self.variant]] = self.data[
                CATEGORICAL_COLUMNS[self.dataset][self.variant]
            ].astype("category")
            self.data[BOOL_COLUMNS[self.dataset]] = self.data[
                BOOL_COLUMNS[self.dataset]
            ].astype("bool")


class NoisyDataset(IIDDataset):
    def __init__(
        self,
        dataset: Literal["BankAccountFraud", "FolkTables"],
        variant: str,
        noise_rates: dict[int, float],
        y_dependant: list[int],
        label_column: Optional[str] = None,
        sensitive_column: Optional[str] = None,
    ):
        """
        Generate a Noisy version of the IID version of the specified Aequitas Dataset.

        Parameters
        ----------
        dataset : Literal["BankAccountFraud", "FolkTables"]
            The Aequitas Dataset.
        variant : str
            The variant of the dataset.
        noise_rates : dict[int, float]
            The noise rates to be applied to each of the sensitive groups.
            Ex: {0: 0.05, 1: 0.1}
        y_dependant : list[int]
            The classe(s) to be affected by the noise. Either [0], [1] or [0, 1].
        label_column : Optional[str], optional
            The label column of the dataset. By default, None, which will use the
            dataset's default label column.
        sensitive_column : Optional[str], optional
            The sensitive column of the dataset. By default, None, which will use the
            dataset's default sensitive column.
        """
        super().__init__(
            dataset=dataset,
            variant=variant,
            label_column=label_column,
            sensitive_column=sensitive_column,
        )
        self.noise_rates = noise_rates
        self.y_dependant = y_dependant

    def load_data(self) -> None:
        """Load the defined dataset."""
        super().load_data()

        noisy_labels = pd.read_csv(
            f"data/{self.dataset}/{self.variant}/noisy/"
            f"label_{self.y_dependant[0] if len(self.y_dependant) == 1 else 'both'}/"
            f"train_{self.noise_rates[0]}_{self.noise_rates[1]}.csv",
            index_col=0,
        )["0"]

        self.data.loc[noisy_labels.index, self.label_column] = noisy_labels
