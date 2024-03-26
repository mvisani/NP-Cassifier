from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def calculate_metrics(
    label: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
    assert isinstance(label, np.ndarray)
    assert isinstance(prediction, np.ndarray)
    assert label.shape == prediction.shape
    return {
        "accuracy": accuracy_score(label, prediction),
        "f1": f1_score(label, prediction),
        "mcc": matthews_corrcoef(label, prediction),
    }


def get_var_name(var):
    for name, value in globals().items():
        if value is var:
            return name
