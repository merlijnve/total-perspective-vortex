from joblib import load

from config import MODELS_PATH
from config import experiments


def load_models():
    models = {}
    for ex in experiments:
        print("Loading model", ex["name"], "...")
        model = load(MODELS_PATH + ex["name"] + "_109_subjects.joblib")
        models[ex["name"]] = model
    return models


def main():
    models = load_models()
    print(models)


if __name__ == "__main__":
    main()
