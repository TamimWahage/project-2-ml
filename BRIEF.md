![Ironhack logo](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project 2 | Machine Learning: From Data to Predictions

*Data Science & Machine Learning — Week 6*

Build a supervised machine learning model that a business could actually use. You will frame a prediction problem, prepare the data, compare models honestly, evaluate the winner once, explain what drives it, and demo it live.

`Python` · `scikit-learn` · `Pipelines` · `Cross-validation` · `Model evaluation` · `Keras (image option)`

---

## Overview

In this project you take one dataset from raw files to a saved model that predicts for new data, following the workflow template from Week 5. The point is not the highest score in the class. It is a model whose score you can trust, chosen for reasons you can explain, and judged against a baseline.

- **Format** — in pairs or on your own. Working in a pair, divide the work by stage, not by model, and run each other's notebooks every day.
- **Duration** — launch on Friday of Week 5, 4 working days, presentations on Friday.
- **Tools** — Python, pandas, scikit-learn, Matplotlib/Seaborn, joblib, Jupyter, GitHub. Keras for the image option.
- **Repository** — click **Use this template** at the top of this page to create your own repo.

### The goal

Answer one question with a model: *who will leave, what is this house worth, whom should we call, which flower is this?* By Friday you should be able to say how good your model is, compared with what, for whom, and where it should not be trusted.

> [!IMPORTANT]
> Choose your dataset and your target on launch day and stick with them. By Wednesday your pipeline and your cross-validation are built around your columns; switching costs four days of work in two.

<br>

## The workflow at a glance

Your project follows the eleven steps of **The ML Workflow End to End** from Week 5. The brief, the run sheets and the rubric all use them:

1. **Frame** — the target, the positive class, who uses the prediction, and the metric, chosen from the cost of each mistake.
2. **Look** — types, missing values, duplicates, the target's distribution, columns that would not be known at prediction time.
3. **Split once** — a fixed `random_state`, stratified for classification. The test set is not touched again until step 9.
4. **Baseline** — a `DummyClassifier` or `DummyRegressor`, scored with cross-validation. Every model has to beat it.
5. **Pipeline** — a `ColumnTransformer` for each column type, then the model. Everything that learns from data goes inside.
6. **Compare** — at least three model families, the same cross-validation folds, the mean and the spread.
7. **Tune** — a grid or random search on the best one or two, on the training set.
8. **Threshold** — for classification, chosen from cross-validated probabilities for a stated reason.
9. **Test once** — the final model on the test set, compared with the cross-validated estimate.
10. **Interpret** — what drives the predictions, and where the model goes wrong.
11. **Save** — the fitted pipeline with `joblib`, reloaded to predict new rows.

The three notebooks in this template follow that order: `01_eda_and_baseline.ipynb` is steps 1–4, `02_pipelines_and_models.ipynb` is steps 5–8, and `03_evaluation_and_interpretation.ipynb` is steps 9–11.

<br>

## Choose your data

Three tabular datasets forming a difficulty ladder, plus an image option. Read [`data/README.md`](data/README.md) for the full description of each, the traps to watch for, and its licence.

| | Dataset | Task | Difficulty | Licence |
|---|---|---|---|---|
| **1** | **King County house sales** | Regression: the sale price | Gentler start | Public domain |
| **2** | **Telco customer churn** | Classification: who leaves | Middle | Apache 2.0 |
| **3** | **Bank Marketing** (UCI) | Imbalanced classification: who subscribes | Hardest | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **4** | **Flower photos** | Image classification: five flowers | Image option | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |

Fetch whichever you choose with the included script:

```bash
python download_data.py houses    # or: telco, bank, flowers
```

Files land in `data/raw/`, which is gitignored. **Do not commit data.** Anyone cloning your repo runs the same script.

> [!NOTE]
> **Bringing your own dataset?** You may, and you are graded on exactly the same rubric. It needs a clear target, at least a few thousand rows (or a few hundred images per class), features that would all be known at prediction time, and a licence that permits educational use. Clear it with your teacher on launch day, not on Wednesday.

### The image option

Option 4 is for the material from Week 5 Thursday: a CNN trained from scratch, then transfer learning with a pretrained network. The workflow is the same, with three differences:

- **The baseline** is the majority class, then a small CNN from scratch; the real model is a pretrained network (for example `MobileNetV2`) with a new output layer.
- **The pipeline lives in the model.** Resizing, rescaling and augmentation are Keras layers, and `image_dataset_from_directory` reads the folders. There is no `ColumnTransformer`.
- **Validation is a fixed split**, not cross-validation: training, validation and test sets chosen once with a seed. Training on a CPU makes repeated folds too slow.

It needs TensorFlow (`python -c "import tensorflow"`) and the download is 218 MB. Everything else in this brief applies.

<br>

## Day-by-day breakdown

Launch day (Friday of Week 5) is when you settle who you are working with, pick your dataset and target, create your repo, download the data and write your framing — so Monday starts with the data in front of you.

### Day 1 — EDA and baseline

- Finish **`01_eda_and_baseline.ipynb`**: shape and types, data quality, the target, the features against the target.
- **Find the columns that would not be known at prediction time** and drop them, with the reason written down.
- **Split once** and score the **baseline** with cross-validation on the training set.

> [!TIP]
> By the end of today you should be able to say your metric and your baseline score out loud. If you cannot, you are behind.

### Day 2 — Pipelines and features

Today is the core engineering day, and the one the week turns on.

- Build **one complete `Pipeline`**: a `ColumnTransformer` that imputes, scales and encodes each type of column, then a model.
- Score it with **cross-validation** against the baseline.
- Add features **inside the pipeline** — a ratio, an age, a flag, a log — and keep the ones that improve the cross-validated score.
- *Image option:* load the folders with `image_dataset_from_directory`, add rescaling and augmentation layers, and train a CNN from scratch with early stopping. Plot the training curves.

> [!IMPORTANT]
> **Anyone without a pipeline scored with cross-validation on Tuesday evening has no Wednesday.** Everything after this builds on it.

### Day 3 — Compare, tune, choose

- Compare **at least three model families** — for example a linear model, a tree-based model and an ensemble — with the same folds, in **`02_pipelines_and_models.ipynb`**.
- **Tune** the best one or two with `GridSearchCV` or `RandomizedSearchCV`, on the training set.
- For classification, **choose the threshold** from cross-validated probabilities.
- Choose your final model, and say why — including when a simpler model is within the noise of the best one.
- *Image option:* transfer learning — a frozen pretrained network with a new output layer, against your CNN from scratch. Fine-tuning is optional.

### Day 4 — Test once, interpret, communicate

- In **`03_evaluation_and_interpretation.ipynb`**, evaluate the final model **once** on the test set.
- Analyse the errors: the confusion matrix, or the residuals, and where the model is worst.
- Explain what drives the predictions with **permutation importance**, and read it with care.
- **Save the fitted pipeline** with `joblib` and reload it to predict new rows: this is your demo.
- Write your README, build your slides, and **rehearse once, timed**.

> [!NOTE]
> **Notebook 03 is your report.** It is read on its own, by someone who has not seen notebooks 01 and 02: the question, the evidence, what you concluded and how far to trust it.

<br>

## Deliverables

Your main deliverable is your GitHub repo, created from this template, containing:

| Item | Description |
|---|---|
| `README.md` | Your project: the question, the data and its licence, the metric, the models compared, the final result against the baseline, and how to reproduce it. Replace this brief with your own. |
| `notebooks/01_eda_and_baseline.ipynb` | Framing, EDA, the split and the baseline. |
| `notebooks/02_pipelines_and_models.ipynb` | Pipelines, features, the model comparison, tuning and the threshold. |
| `notebooks/03_evaluation_and_interpretation.ipynb` | **The report.** The test-set evaluation, the error analysis, the interpretation, and saving and reloading the model. |
| `src/functions.py` | Reusable functions. Your logic lives in functions, not pasted into cells. |
| `download_data.py` | Left as it is, or extended if you brought your own data. |
| Slides | Linked or committed in your README, so they can be read after the presentation. Any tool. |

The saved model in `models/` is gitignored: notebook 03 recreates it, so anyone can reproduce it.

### Minimum requirements

Your project must meet all of these to pass:

- **Framing** — a stated target, positive class, user and decision, and a metric chosen from the cost of each mistake.
- **No leakage** — columns not known at prediction time dropped; the test set used once; every step that learns from data inside the pipeline.
- **Baseline** — scored with the same cross-validation as the models.
- **Comparison** — at least **3 model families** on the same folds, with the mean and spread. *Image option:* a CNN from scratch and a transfer-learning model.
- **Tuning** — the best one or two tuned on the training set.
- **Evaluation** — the final model evaluated once on the test set, with an error analysis and, for classification, a justified threshold.
- **Interpretation** — what drives the predictions, explained.
- **A saved model** — reloaded to predict new data, in the notebook and in the demo.

The full grading rubric is in [`RUBRIC.md`](RUBRIC.md). Read it on launch day — it tells you exactly what is being evaluated.

<br>

## Advanced features (optional)

Not required, but they will strengthen your project:

- **Learning curves** — does more data help? Train on growing fractions and plot the cross-validated score.
- **Stacking** — a `StackingClassifier` or `StackingRegressor` over your best models, compared fairly with them.
- **Segmented evaluation** — the error by group (by contract type, by price band, by job), and where the model should not be used.
- **Cost-based threshold** — put a money value on each kind of mistake and choose the threshold that minimises the total.
- **A tiny app** — a Streamlit page that loads your saved model and predicts from a form or an uploaded photo.
- *Image option:* fine-tune the top of the pretrained network after training the new layer, or compare two pretrained networks.

<br>

## Coding best practices

- **Modularise.** Python logic in `src/functions.py` with reusable functions. The notebook is for the narrative.
- **One seed.** Define `RANDOM_STATE` once and pass it everywhere, so a second run gives the same numbers.
- **Name clearly.** Descriptive names for variables, functions and columns. `snake_case` throughout.
- **Clean up.** Remove unused imports, commented-out code and test cells before submitting.
- **Restart and run all.** Before you commit a notebook, run it top to bottom on a fresh kernel.
- **Commit often.** Small, frequent commits with descriptive messages. Working in a pair, your partner should know what you worked on from the history alone.

<br>

## Presentation guidelines

| Component | Duration |
|---|---|
| Talking with slides | 7 minutes |
| Live demo | 3 minutes |
| **Total** | **10 minutes** |

> [!IMPORTANT]
> **You present from your own machine by sharing your screen.** Use whatever slide tool you prefer. The demo loads your saved model and predicts for new data: a customer, a house, a client, a photo. Have it open and ready before your slot.

### Suggested slide structure (~10 slides)

1. **Title** — project title and your name or names.
2. **The problem** — what you predict, for whom, and what they would do with it.
3. **The data** — source and licence, what one row is, the traps you found.
4. **The metric and the baseline** — why this metric, and the score to beat.
5. **What you tried** — the model comparison, with the spread.
6. **The final model** — the test-set result against the baseline and the cross-validated estimate.
7. **Where it goes wrong** — the error analysis.
8. **What drives it** — the interpretation, in business terms.
9. **Conclusions and limitations** — would you use it, and where not?
10. **Closing** — project title, your name or names, thank you.

<br>

## Repo layout

The notebooks are laid out in sections with the intent of each one written down, and `src/functions.py` holds a few empty stubs that exist only to show the shape. **This is a place to start, not a template to fill in** — rename things, add sections, delete the ones that do not fit your data. You are graded on the work, not on how closely you followed the scaffold.

```
.
├── README.md                                    this brief — replace it with your own
├── RUBRIC.md                                    how you are graded
├── requirements.txt
├── download_data.py                             fetches your chosen dataset into data/raw/
├── data/
│   ├── raw/                                     downloaded data, gitignored
│   └── README.md                                the four options, their traps, licences
├── models/                                      the saved model, gitignored
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb                frame, look, split, baseline
│   ├── 02_pipelines_and_models.ipynb            pipelines, compare, tune, threshold
│   └── 03_evaluation_and_interpretation.ipynb   test once, interpret, save, report
└── src/
    └── functions.py                             your reusable logic
```

## Getting set up

```bash
# 1. Click "Use this template" above, then clone YOUR new repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Check you already have the libraries - a current Anaconda install does.
#    If this prints, skip to step 3 and install nothing.
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, joblib; print('all present')"

#    Image option only: TensorFlow is not part of Anaconda.
python -c "import tensorflow as tf; print(tf.__version__)"

#    Only if a check failed. See the note in requirements.txt first: running pip
#    against a conda environment can downgrade packages you did not ask about.
pip install -r requirements.txt

# 3. Download your dataset
python download_data.py telco        # or: houses, bank, flowers

# 4. Open the first notebook
jupyter lab notebooks/01_eda_and_baseline.ipynb
```

<br>

## Tips for success

1. **Choose your data on launch day and commit to it.** Switching mid-week costs you time you do not have.
2. **Beat the baseline before you beat anyone else.** A model that cannot beat a dummy is not a model.
3. **Never look at the test set until Thursday.** Every peek makes your final number a little less true.
4. **Put everything that learns inside the pipeline.** A scaler fitted before the split is the most common way to fool yourself.
5. **Prefer a simple model you understand** when a complex one is within the noise.
6. **Commit early and often.** In a pair, so your partner knows what changed; on your own, so you do.
7. **Ask for help early.** Stuck for more than 30 minutes? Reach out to a classmate, your TA or your teacher.
8. **Practise the presentation out loud** at least once before Friday. Time yourselves.
9. **Read [`RUBRIC.md`](RUBRIC.md).** It tells you exactly what is being evaluated.

Good luck. Make something you would trust.
