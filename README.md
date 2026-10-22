# Bank Marketing: who subscribes to a term deposit?

*Project 2 | Machine Learning: From Data to Predictions* · Ironhack DSML

**Team:** Tamim Wahage
**Slides:** [PDF](presentation/Bank%20Marketing%20Who%20Subscribes.pdf) · [PowerPoint](presentation/Bank%20Marketing%20Who%20Subscribes.pptx)

In this project a Portuguese bank calls its clients to sell them a term deposit, and most calls end with a no. I built a model that predicts *before the call* whether a client will subscribe, so the marketing team can call the likely ones first and skip the ones who will almost surely say no.

The original project brief is in [BRIEF.md](BRIEF.md), the grading rubric in [RUBRIC.md](RUBRIC.md).

## The question

- **What is predicted:** will this client subscribe to a term deposit? Target `y`, positive class `yes`
- **For whom:** the marketing team that plans the call list
- **What they do with it:** call the clients with a high probability first and skip the rest
- **Metric:** only 11.3% of the clients said yes, so accuracy is useless (always saying "no" is already 88.7% accurate). A missed yes is a lost sale and worse than a wasted call, so the models are compared with **F1**, and recall is checked as well
- **Baseline:** calling everyone ("always yes") gets **F1 = 0.203**

## Result

The final model is a `HistGradientBoostingClassifier` with `class_weight="balanced"`, default settings and a decision threshold of **0.60**.

| | F1 | Recall | Precision |
|---|---|---|---|
| **Final model, test set** (used once) | **0.519** | 0.609 | 0.453 |
| Final model, cross-validation | 0.496 ± 0.012 | – | – |
| Baseline: call everyone | 0.203 | 1.000 | 0.113 |

With 15% of the calls the model reaches 61% of the subscribers. Calling everyone reaches all of them, but 89% of those calls are wasted.

**But:** the model mostly recognises *when* a call happened, not *who* the client is. The economy columns (`nr.employed`, `euribor3m`, …) work like a hidden date. In the bad economic periods, where almost half of the clients said yes, the model calls nearly everyone. In the good periods it calls almost nobody. When I trained on the older calls and tested on the newer ones, it got an F1 of 0.398, worse than just calling everyone (0.468). So the score holds for clients from the same periods as the training data, not for a future campaign.

What the bank can use right away are the patterns: clients with a success in the last campaign say yes about 65% of the time, clients with a mobile number, students and retired people subscribe more often, and calling the same client again and again rarely works.

The full report is [notebook 03](notebooks/03_evaluation_and_interpretation.ipynb).

## Data

The data is the **Bank Marketing** dataset from the UCI Machine Learning Repository:

> Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5K306>. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

It has 41,188 phone contacts from May 2008 to November 2010. One row is one call to one client, with the client's data (age, job, loans, …), earlier contacts and the economy at that time. More about the columns is in [data/README.md](data/README.md).

What I cleaned (see [notebook 01](notebooks/01_eda_and_baseline.ipynb)):

- `duration` (the length of the call) dropped: it's only known after the call and gives the answer away
- 12 exact duplicate rows removed
- `unknown` kept as its own category (it's the hidden missing value in six columns)
- `pdays` = 999 means "never contacted": replaced by a yes/no "contacted before" flag
- `housing` and `loan` dropped: no relationship with the target, and the same F1 without them

## Models compared

All four model families on the same 5 stratified folds of the training set (80%), each with six ways of handling the imbalance: none, class weights, random oversampling, SMOTE, random undersampling and cluster centroids (undersampling by k-means clustering). Each combination is scored at its own best threshold, because the handling mostly just moves that threshold. Best version of each family:

| Model | Imbalance handling | F1 (mean ± spread) |
|---|---|---|
| **Gradient boosting** | **class weights** | **0.499 ± 0.014** |
| Logistic regression | random oversampling | 0.485 ± 0.012 |
| Random forest | random undersampling | 0.466 ± 0.014 |
| Decision tree | SMOTE | 0.338 ± 0.014 |
| Baseline: call everyone | | 0.203 |

The model family matters, the imbalance method hardly does: for gradient boosting and logistic regression every method except cluster centroids ends up within the fold spread. Cluster centroids drops gradient boosting to baseline level. I kept class weights: just as good, and no clients copied, made up or thrown away. Gradient boosting beats logistic regression on every one of the 5 folds.

Tuning (grid search for logistic regression, random search for gradient boosting) didn't beat the default settings. The threshold was chosen from cross-validated probabilities: 0.60 is the lowest threshold within the fold spread of the best F1, because a missed yes costs more than a wasted call. Details are in [notebook 02](notebooks/02_pipelines_and_models.ipynb).

## How to run

```bash
git clone <this-repo-url>
cd project-2-ml

# check the libraries first, a current Anaconda install has them
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, joblib; print('all present')"
python -c "import imblearn"          # only needed for the imbalance comparison in notebook 02

python download_data.py bank         # downloads the data into data/raw/

jupyter lab notebooks/
```

Then run the notebooks in this order:

1. `01_eda_and_baseline.ipynb` – framing, exploring the data, the split and the baseline
2. `02_pipelines_and_models.ipynb` – pipeline, features, model comparison, tuning and the threshold
3. `03_evaluation_and_interpretation.ipynb` – the test set, errors, what drives the model, and saving it

Notebook 03 saves the fitted model to `models/bank_marketing_model.joblib` (gitignored) and reloads it to predict new clients. All the reusable code (loading, cleaning, the split, the pipeline, the final model) is in [src/functions.py](src/functions.py), so every notebook uses exactly the same rows and the same model.

## Limitations

- The split is random, so training and test data come from the same time periods. On newer calls the model did worse than calling everyone.
- The data is from one Portuguese bank during the financial crisis (2008–2010). It probably doesn't hold for other banks, countries or a different economy.
- The threshold is chosen with F1, not with real money. The cost of a call and the value of a term deposit would make it a business decision.
- In "good" economic periods the model calls nobody and misses almost every subscriber there.
- With another week I would validate by time (train on older, test on newer), try a model without the economy columns, and retrain it on the most recent campaigns.
