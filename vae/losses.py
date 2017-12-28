import numpy as np

def squared_error(y, yhat):
    return y - yhat

def identity(y, yhat):
    return yhat

loss_table = {
    'squared_error': squared_error,
    'identity': identity
}