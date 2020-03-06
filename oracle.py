import numpy as np

from .Probleme_R import *
from .Structures_N import *

def OraclePG(qc, ind):
    if ind == 2:
        return (1/3)*np.dot(q0 + np.dot(B, qc), r*(q0 + np.dot(B, qc))*np.abs(q0 + np.dot(B, qc))) + np.dot(pr, np.dot(Ar, q0 + np.dot(B, qc)))
    if ind == 3:
        q = q0 + np.dot(B, qc)
        return np.dot(B.T, r*q*np.abs(q)) + np.dot(np.dot(B.T, Ar.T), pr)

