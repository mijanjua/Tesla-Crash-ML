import pandas as pd
import pytest

from src import data, train


@pytest.fixture(scope="module")
def split():
    return train.split(data.load_dataset())


def test_split_is_stratified(split):
    _, _, y_train, y_test = split
    assert abs(y_train.mean() - y_test.mean()) < 0.03


@pytest.mark.parametrize("name", ["Logistic Regression", "Random Forest"])
def test_models_fit_and_predict_probabilities(split, name):
    X_train, X_test, y_train, _ = split
    model = train.build_models()[name].fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    assert proba.shape == (len(X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_model_handles_unseen_category(split):
    X_train, X_test, y_train, _ = split
    model = train.build_models()["Logistic Regression"].fit(X_train, y_train)
    row = X_test.iloc[[0]].copy()
    row["model"] = "Cybertruck"  # never seen in training
    assert model.predict_proba(row).shape == (1, 2)


def test_evaluate_on_test_returns_confusion_matrix(split):
    X_train, X_test, y_train, y_test = split
    model = train.build_models()["Logistic Regression"].fit(X_train, y_train)
    result = train.evaluate_on_test(model, X_test, y_test)
    cm = result["confusion_matrix"]
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y_test)
    assert 0 <= result["roc_auc"] <= 1


def test_importances_cover_all_encoded_features(split):
    X_train, _, y_train, _ = split
    models = train.build_models()
    lr = models["Logistic Regression"].fit(X_train, y_train)
    rf = models["Random Forest"].fit(X_train, y_train)
    coefs = train.logistic_coefficients(lr)
    imps = train.forest_importances(rf)
    assert isinstance(coefs, pd.Series) and len(coefs) == len(imps)
    assert set(data.NUMERIC_FEATURES) <= set(coefs.index)
    assert imps.sum() == pytest.approx(1.0)
