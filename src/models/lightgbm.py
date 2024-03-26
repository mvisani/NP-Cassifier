from hyperopt import hp
from .abstract_model import AbstractModel
import numpy as np
from typing import Optional
from lightgbm import LGBMClassifier
from dict_hash import sha256
import compress_pickle


class LightGBM(AbstractModel):
    def __init__(
        self,
        boosting_type: str,
        num_leaves: int,
        max_depth: int,
        # learning_rate: float,
        n_estimators: int,
        importance_type: str,
    ) -> None:
        super().__init__()
        params = dict(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            n_estimators=n_estimators,
            importance_type=importance_type,
            random_state=83495,
            n_jobs=-1,
        )
        self.model = LGBMClassifier(
            **params,
        )
        self.params = params

    def search_space():
        return {
            "boosting_type": hp.choice("boosting_type", ["gbdt", "dart"]),
            "num_leaves": hp.uniformint("num_leaves", 2, 100),
            "max_depth": hp.uniformint("max_depth", -1, 100),
            "n_estimators": hp.choice(
                "n_estimators", [10, 20, 50, 100, 200, 300, 1000]
            ),
            "importance_type": hp.choice("importance_type", ["split", "gain"]),
        }

    @staticmethod
    def from_params(params):
        return LightGBM(
            boosting_type=params["boosting_type"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            importance_type=params["importance_type"],
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None
    ) -> None:
        self.model.fit(X, y, sample_weight=weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def dump_model(self, path: str) -> None:
        compress_pickle.dump(self, path)

    @staticmethod
    def load_model(path: str):
        return compress_pickle.load(path)

    def consistent_hash(self) -> str:
        return sha256(
            {
                **self.params,
                "model": "Lightgbm",
            }
        )
