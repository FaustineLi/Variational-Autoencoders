import math
import random

def act(x):
    mat = list()
    for r in x:
        row = list()
        for c in r:
            row.append(1 / (1 + math.exp(- c)))
        mat.append(row)
    return mat
    
def grad_act(x):
    mat = list()
    for r in x:
        row = list()
        for c in r:
            row.append(math.exp(- c) / (1 + math.exp(- c))**2)
        mat.append(row)
    return mat

def loss(x):
    mat = list()
    for r in x:
        row = list()
        for c in r:
            row.append(math.exp(- c) / (1 + math.exp(- c))**2)
        mat.append(row)
    return mat

def grad_loss(A, B):
    mat = list()
    for i, row in enumerate(A):
        r = list()
        for j, column in enumerate(row):
            r.append( -A[i][j] + B[i][j])
        mat.append(r)
    return mat