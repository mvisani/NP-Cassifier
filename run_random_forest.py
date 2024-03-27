import pandas as pd
from hpsklearn import random_forest_classifier

from src.run import run


def main():
    dataset = pd.read_excel("data/NPClassifier_dataset.xlsx")
    run(
        dataset=dataset,
        model=random_forest_classifier,
        model_name="random_forest_classifier",
        holdout=10,
        max_evals=20,
        test_size=0.2,
        n_jobs=-1,
        verbose=True,
    )

if __name__ == "__main__":
    main()