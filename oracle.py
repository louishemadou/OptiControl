import numpy as np

from Problemes_R import *
from Structures_N import *


def OraclePG(qc):
    q = q0 + np.dot(B, qc)
    F = (1/3)*np.dot(q, r*q*np.abs(q)) + np.dot(pr, np.dot(Ar, q))
    G = np.dot(B.T, r*q*np.abs(q)) + np.dot(np.dot(B.T, Ar.T), pr)
    return F, G


def OraclePH(qc):
    q = q0 + np.dot(B, qc)
    [F, G, ind] = OraclePG(qc)
    H = 2*np.dot(B.T, np.dot(np.diag(r*np.abs(q)), B))
    return F, G, H
