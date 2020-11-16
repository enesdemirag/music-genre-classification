import pandas as pd

def test_model(model, features, labels):
    _, acc = model.evaluate(features, labels, verbose=0)
    return acc

def predict(model, features):
    output = model(features)
    return output