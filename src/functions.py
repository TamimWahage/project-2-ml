"""Your reusable functions live here.

The rule from the brief: Python logic goes in `.py` files, and the notebooks
hold the narrative. The moment a cell grows past a few lines, or you find
yourself pasting it a second time, move it here and call it from the notebook.

To use this module from a notebook in `notebooks/`:

    import sys
    sys.path.append("..")
    from src.functions import *

The four below are only there to show the shape. Rename them, change the
arguments, write your own. This is a starting point, not an interface you have
to implement.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

RANDOM_STATE = 42


def load_data():
    """Read your raw file and apply the fixes that do not learn from the data."""
    import pandas as pd

    return pd.read_csv(RAW / "bank_marketing.csv", sep=";")


def clean_data(df):
    """Drop the exact duplicates and `duration`, and turn `y` into 1/0."""
    # duplicates first: without `duration` many different calls would look the same
    df = df.drop_duplicates().drop(columns="duration")
    df["y"] = (df["y"] == "yes").astype(int)
    return df


def split_data(df, test_size=0.2):
    """The one train/test split every notebook uses, stratified on `y`."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns="y")
    y = df["y"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def cramers_v_table(df, columns, target="y"):
    """Chi-square p-value and Cramér's V of each categorical column against the target."""
    import pandas as pd
    from scipy.stats import chi2_contingency
    from scipy.stats.contingency import association

    rows = []
    for col in columns:
        crosstab = pd.crosstab(df[col], df[target])
        rows.append({"column": col,
                     "p_value": chi2_contingency(crosstab)[1],
                     "cramers_v": association(crosstab, method="cramer")})
    return pd.DataFrame(rows).set_index("column").sort_values("cramers_v", ascending=False)


def plot_rate_by(df, columns, target="y", ncols=2):
    """Bar chart of the share of positives per category, one panel per column."""
    import matplotlib.pyplot as plt

    nrows = -(-len(columns) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    overall = df[target].mean()
    for ax, col in zip(axes.flat, columns):
        rates = df.groupby(col, observed=True)[target].mean().sort_values()
        ax.barh(rates.index.astype(str), rates.values, color="#4C72B0")
        ax.axvline(overall, color="gray", linestyle="--", linewidth=1)
        ax.set_title(col)
        ax.set_xlabel("share of yes")
    for ax in axes.flat[len(columns):]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.show()


NUMERIC = ["age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
           "cons.conf.idx", "euribor3m", "nr.employed"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact",
               "month", "day_of_week", "poutcome"]


def add_features(X, age_groups=False, contacted_flag=False, one_economy=False, drop=()):
    """The feature ideas from notebook 01. None of them learns from the data."""
    import pandas as pd

    X = X.copy()
    if age_groups:
        X["age_group"] = pd.cut(X["age"], [0, 25, 30, 40, 50, 60, 120]).astype(str)
    if contacted_flag:
        X["contacted_before"] = (X["previous"] > 0).astype(int)
        X = X.drop(columns="pdays")
    if one_economy:
        X = X.drop(columns=["emp.var.rate", "nr.employed"])     # keep euribor3m
    return X.drop(columns=list(drop))


def build_preprocessor(numeric, categorical):
    """Return a ColumnTransformer that prepares each type of column."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


def build_pipeline(model, **features):
    """Features, preprocessing and model in one object, so raw rows go in."""
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    # run the feature step on an empty frame to know which columns come out of it
    columns = add_features(pd.DataFrame(columns=NUMERIC + CATEGORICAL), **features).columns
    numeric = [c for c in columns if c not in CATEGORICAL and c != "age_group"]
    categorical = [c for c in columns if c in CATEGORICAL or c == "age_group"]
    return Pipeline([
        ("features", FunctionTransformer(add_features, kw_args=features)),
        ("prep", build_preprocessor(numeric, categorical)),
        ("model", model),
    ])


def build_smote_pipeline(model, **features):
    """Like build_pipeline, but with SMOTE between preprocessing and model.

    The imbalanced-learn pipeline only oversamples during fit, so every
    validation fold keeps its real share of yes.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    steps = build_pipeline(model, **features).steps
    return ImbPipeline(steps[:2] + [("smote", SMOTE(random_state=RANDOM_STATE))] + steps[2:])


def make_cv():
    """The cross-validation every model is scored with: 5 stratified folds."""
    from sklearn.model_selection import StratifiedKFold

    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def make_scoring():
    """F1 to compare models, plus recall, precision and accuracy to read them."""
    from sklearn.metrics import make_scorer, precision_score

    return {"f1": "f1", "recall": "recall", "accuracy": "accuracy",
            "precision": make_scorer(precision_score, zero_division=0)}   # 0 instead of a warning when nothing is flagged


def compare_models(models, X, y, cv, scoring):
    """Cross-validate each model on the same folds and return a tidy table."""
    import pandas as pd
    from sklearn.model_selection import cross_validate

    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        rows.append({"model": name,
                     "F1": scores["test_f1"].mean(),
                     "F1 std": scores["test_f1"].std(),
                     "recall": scores["test_recall"].mean(),
                     "precision": scores["test_precision"].mean(),
                     "accuracy": scores["test_accuracy"].mean()})
    return pd.DataFrame(rows).set_index("model").round(3)


def plot_comparison(table, baseline):
    """Mean F1 per model with the spread across folds, and the baseline as a line."""
    import matplotlib.pyplot as plt

    table = table.sort_values("F1")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(table) + 1))
    ax.barh(table.index, table["F1"], xerr=table["F1 std"], color="#4C72B0", capsize=3)
    ax.axvline(baseline, color="gray", linestyle="--", linewidth=1, zorder=3)
    ax.text(baseline, len(table) - 0.4, f" baseline {baseline:.3f}", color="gray", va="center")
    ax.set_xlabel("cross-validated F1")
    fig.tight_layout()
    plt.show()


def threshold_table(y_true, proba, thresholds):
    """F1, recall, precision, missed yes and wasted calls for each threshold."""
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score

    rows = []
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        rows.append({"threshold": round(threshold, 2),
                     "F1": f1_score(y_true, pred),
                     "recall": recall_score(y_true, pred),
                     "precision": precision_score(y_true, pred, zero_division=0),
                     "calls": int(pred.sum()),
                     "missed yes": int(((pred == 0) & (y_true == 1)).sum()),
                     "wasted calls": int(((pred == 1) & (y_true == 0)).sum())})
    return pd.DataFrame(rows).set_index("threshold").round(3)


FINAL_FEATURES = dict(contacted_flag=True, drop=("housing", "loan"))
FINAL_THRESHOLD = 0.6


def final_pipeline():
    """The model chosen in notebook 02, not fitted yet."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    return build_pipeline(model, **FINAL_FEATURES)


def f1_at_threshold(threshold):
    """A scorer for F1 at our own threshold instead of the default 0.5."""
    from sklearn.metrics import f1_score

    def scorer(estimator, X, y):
        return f1_score(y, estimator.predict_proba(X)[:, 1] >= threshold)
    return scorer


def cv_scores_at_threshold(make_model, X, y, cv, threshold):
    """F1 on every fold, with the model refitted on the other folds."""
    import numpy as np

    scorer = f1_at_threshold(threshold)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        model = make_model().fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(scorer(model, X.iloc[val_idx], y.iloc[val_idx]))
    return np.array(scores)


def plot_errors(y_true, y_pred):
    """Show where the model goes wrong: a confusion matrix or the residuals."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["no", "yes"],
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    plt.show()


def segment_table(results, column):
    """Per group: how many clients, how many yes, and how the model does there."""
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score

    def scores(group):
        return pd.Series({"clients": len(group),
                          "yes": group["y"].sum(),
                          "share called": group["pred"].mean(),
                          "recall": recall_score(group["y"], group["pred"], zero_division=0),
                          "precision": precision_score(group["y"], group["pred"], zero_division=0)})
    return results.groupby(column, observed=True).apply(scores, include_groups=False).round(3)
