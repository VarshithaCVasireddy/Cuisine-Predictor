from types import NoneType
import pytest

from predictor import *


def test_pre_process_ingredients():
    actual_ingredients = pre_process_ingredients(["Rices flour", "scrambled eggs"])
    assert actual_ingredients == "rice_flour,scrambled_egg"

@pytest.mark.skip(reason="time taking test")
def test_get_yummly_data():
    actual_x, actual_y = get_yummly_data()
    assert actual_x.shape == (39774, 6880)
    assert actual_y.shape == (39774,)

@pytest.fixture(scope="module")
def models():
    svc, nn = fit_prediction_models()

    return svc, nn

def test_fit_prediction_models(models):
    assert models[0] is not NoneType
    assert models[1] is not NoneType

def test_find_cuisine(models):
    actual_cuisine, actual_score = find_cuisine(models[0], ["Rices flour", "scrambled eggs"])
    
    assert actual_cuisine == "indian"
    assert actual_score == 0.36

def test_find_closest(models):
    actual_closest = find_closest(models[1], ["Rices flour", "scrambled eggs"], 5)
    
    assert len(actual_closest) == 5
