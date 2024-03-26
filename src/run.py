import multiprocessing
from functools import partial
from typing import Optional

import pandas as pd
from cache_decorator import Cache
from deflate_dict import deflate
from hpsklearn import HyperoptEstimator
from hyperopt import tpe
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.mol_to_fp import calculate_fingerprint


@Cache(
    cache_dir="experiments/n_holdouts_{holdout}/{model_name}/{_hash}",
    cache_path="{cache_dir}/performanc.csv",
)
def run(
    dataset: pd.DataFrame,
    model,
    model_name: Optional[str],
    holdout: int,
    max_evals: int,
    test_size: float,
    n_jobs: int,
    verbose=True,
):
    pool = multiprocessing.Pool()
    X = list(
        tqdm(
            pool.imap(
                partial(calculate_fingerprint, radi=2),
                dataset["index"].values,
                chunksize=1000,
            ),
        )
    )
    y = dataset["Class"].values

    result = []
    for i in tqdm(range(holdout), desc="Holdout number"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=(i + 324) * 5723
        )

        estim = HyperoptEstimator(
            classifier=model(f"model_{i}"),
            preprocessing=[],
            algo=tpe.suggest,
            max_evals=max_evals,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        estim.fit(X_train, y_train)
        y_pred = estim.best_model()["learner"].predict(X_test)

        tmp = deflate(estim.best_model()["learner"].get_params())
        tmp["holdout"] = i
        tmp["max_evals"] = max_evals
        tmp["test_size"] = test_size
        tmp["model_name"] = model_name
        tmp["accuracy"] = estim.score(X_test, y_test)
        tmp["mcc"] = matthews_corrcoef(y_test, y_pred)
        tmp["f1_score"] = f1_score(y_test, y_pred, average="macro")

        result.append(tmp)

    return pd.DataFrame(result)
