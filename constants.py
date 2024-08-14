from aequitas.flow.datasets import BankAccountFraud, FolkTables
from aequitas.flow.datasets.baf import (
    VARIANTS as BAF_VARIANTS,
    SENSITIVE_COLUMN as BAF_SENSITIVE_COLUMN,
    CATEGORICAL_COLUMNS as BAF_CATEGORICAL_COLUMNS,
    LABEL_COLUMN as BAF_LABEL_COLUMN,
)
from aequitas.flow.datasets.folktables import (
    VARIANTS as FT_VARIANTS,
    SENSITIVE_COLUMN as FT_SENSITIVE_COLUMN,
    CATEGORICAL_COLUMNS as FT_CATEGORICAL_COLUMNS,
    LABEL_COLUMNS as FT_LABEL_COLUMNS,
    BOOL_COLUMNS as FT_BOOL_COLUMNS,
)

ABBREVIATIONS = {"BankAccountFraud": "baf", "FolkTables": "acs"}

CLASSES = {"BankAccountFraud": BankAccountFraud, "FolkTables": FolkTables}

NOISE_TYPES = {
    "I": [(0, 0), (5, 5), (10, 10), (20, 20)],
    "II": [(0, 0), (0, 5), (0, 10), (0, 20)],
    "III": [(5, 0), (5, 5), (5, 10), (5, 20)],
}

MARKERS = {0: "*", 1: "o", 2: "s", 3: "v"}

METRICS = {"Demographic Parity": "pprev_ratio", "TPR": "tpr"}

COLORS = {
    "lightgbm": "tab:blue",
    "OBNC": "tab:orange",
    "Fair-OBNC": "tab:green",
    "DataRepairer": "tab:red",
    "PrevalenceSampling": "tab:purple",
    "Massaging": "tab:brown",
    "CorrelationSuppression": "tab:pink",
    "FeatureImportanceSuppression": "tab:gray",
}


VARIANTS = {
    "BankAccountFraud": BAF_VARIANTS,
    "FolkTables": FT_VARIANTS,
}

SENSITIVE_COLUMN = {
    "BankAccountFraud": BAF_SENSITIVE_COLUMN,
    "FolkTables": FT_SENSITIVE_COLUMN,
}

CATEGORICAL_COLUMNS = {
    "BankAccountFraud": BAF_CATEGORICAL_COLUMNS,
    "FolkTables": FT_CATEGORICAL_COLUMNS,
}

LABEL_COLUMN = {
    "BankAccountFraud": BAF_LABEL_COLUMN,
    "FolkTables": FT_LABEL_COLUMNS,
}

BOOL_COLUMNS = {"FolkTables": FT_BOOL_COLUMNS}

NOISE_RATES = [
    (0, 0),
    (5, 5),
    (10, 10),
    (20, 20),
    (0, 5),
    (0, 10),
    (0, 20),
    (5, 0),
    (5, 10),
    (5, 20),
]
