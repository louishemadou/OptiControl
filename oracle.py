import numpy as np

from Problemes_R import *
from Structures_N import *


def OraclePG(qc, ind):
    q = q0 + np.dot(B, qc)
    F = 0
    G = 0
    if ind == 2 or ind == 4:
        F = (1/3)*np.dot(q, r*q*np.abs(q)) + np.dot(pr, np.dot(Ar, q))
    if ind == 3 or ind == 4:
        G = np.dot(B.T, r*q*np.abs(q)) + np.dot(np.dot(B.T, Ar.T), pr)
    return [F, G, ind]
