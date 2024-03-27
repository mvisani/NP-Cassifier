import pandas as pd
from hpsklearn import extra_trees_classifier

from src.run import run


def main():
    dataset = pd.read_excel("data/NPClassifier_dataset.xlsx")
    run(
        dataset=dataset,
        model=extra_trees_classifier,
        model_name="extra_trees_classifier",
        holdout=10,
        max_evals=10,
        test_size=0.2,
        n_jobs=-1,
        verbose=False,
    )

if __name__ == "__main__":
    main()