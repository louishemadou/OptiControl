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

def OraclePH(qc, ind):
    q = q0 + np.dot(B, qc)
    F, G, H = 0, 0, 0
    if ind < 4:
        [F, G, ind] = OraclePG(qc, ind)
    elif ind == 6:
        [F, G, ind] = OraclePG(qc, 3)
    elif ind == 7:
        [F, G, ind] = OraclePG(qc, 4)
    if ind in [5, 6, 7]:
        H = 2*np.dot(B.T, np.dot(np.diag(r*np.abs(q)), B))
    return [F, G, H, ind]
