from BFGS import bfgs
from PR import Polak_Ribiere
from newton import Newton
from oracle import *


def solve(method, visual = False):
    x0 = np.random.normal(size=n - md)
    if method == "BFGS":
        F, x, G = bfgs(OraclePG, x0, visual = visual)
    if method == "Newton":
       F, x, G = Newton(OraclePH, x0, visual = visual)
    if method == "PR":
        F, x, G = Polak_Ribiere(OraclePG, x0, visual = visual)
    return F, x, G

F, x, G = solve("PR", visual = True)

print("Critère optimal = " + str(F))
print("Optimal au point " + str(x))
