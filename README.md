# Tesla Fatal Crashes: Predicting "Autopilot Claimed"

An end-to-end machine learning project that predicts whether Autopilot was **claimed** to be active in a fatal Tesla crash, from when and where the crash happened, the car model, and who was killed.

It covers data cleaning, feature engineering, logistic regression and random forest models, evaluation (cross-validation, confusion matrices, ROC/PR curves, feature importance), and a Streamlit app.

## Project structure

```
data/Tesla Deaths - Deaths.csv   raw dataset
src/data.py                      cleaning + feature engineering (shared by everything)
src/train.py                     models, evaluation, saves the best model
notebooks/01_eda_and_model.ipynb step-by-step walkthrough with charts
app/streamlit_app.py             interactive prediction app
models/                          trained model + metrics used by the app
```

## Quick start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m src.train                 # train both models, save the best
streamlit run app/streamlit_app.py  # open http://localhost:8501
```

## Approach

- **Target:** `Autopilot claimed`. `-` is treated as "no", any count as "yes", and blank rows are excluded as unknown. That leaves 276 labelled crashes, 13% of them positive.
- **Features:** year, month, region, US state group, car model, death counts by victim type (Tesla driver, passengers, other vehicle, cyclists/pedestrians), a multi-fatality flag, and keyword flags from the description.
- **Leakage prevention:** the "Verified Autopilot" columns and description words such as "Autopilot" or "FSD" are excluded, because they reveal the answer.
- **Imbalance:** `class_weight="balanced"`, stratified splits, and models judged on PR-AUC, ROC-AUC, recall and precision rather than accuracy.

## Results

5-fold cross-validation on the training set:

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Baseline (always "no") | 0.50 | 0.13 |
| Logistic Regression | 0.57 | 0.30 |
| Random Forest | 0.57 | 0.24 |

Logistic regression is the deployed model. The signal is weak. The strongest feature is whether the car model was reported at all, which reflects how detailed the news reports were rather than how the car was driven.

## Limitations

- Small dataset: about 35 positive cases, so every metric has wide uncertainty.
- The target records *claims* made in reports, not verified Autopilot use.
- The features describe the crash outcome, not the car's state, such as speed, road type or driver statements.

## Data

The dataset comes from the community-maintained Tesla Deaths spreadsheet (tesladeaths.com).
