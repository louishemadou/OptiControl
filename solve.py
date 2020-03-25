from BFGS import bfgs
from Wolfe_Skel import Newton, Polak_Ribiere
from oracle import *


def solve(method):
    x0 = np.random.normal(size=n - md)
    if method == "BFGS":
        F, x, G = bfgs(OraclePG, x0)
    if method == "Newton":
       F, x, G = Newton(OraclePH, x0)
    if method == "PK":
        F, x, G = Polak_Ribiere(OraclePG, x0)
    return F, x, G


F, x, G = solve("PK")

print("Critère optimal = " + str(F))
print("Optimal au point " + str(x))
