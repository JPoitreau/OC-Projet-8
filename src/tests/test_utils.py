"""Teste les fonctions utilitaires pour le scoring."""

import numpy as np
import pandas as pd

from src.utils.utils import business_cost, custom_sampler_ratio


def test_business_cost():
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = pd.Series([0, 1, 1, 0])

    assert business_cost(y_true, y_pred) == 11


def test_custom_sampler_ratio():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y = pd.Series([0] * 180 + [1] * 20)

    X_res, y_res = custom_sampler_ratio(X, y, undersampler_ratio=0.5)

    assert len(X_res) == len(y_res)
    assert set(y_res.unique()) == {0, 1}
