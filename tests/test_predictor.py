from pandas import notnull
import pytest
import predictor

from predictor import *


def fit_prediction_models():
    assert fit_prediction_models().svc is not None

