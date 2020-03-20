from BFGS import bfgs
from oracle import *

def solve(method):
    if method == "BFGS":
       F, x, G = bfgs(OraclePG, q0)
    return F, x, G

F, x, G = solve("BFGS")

print("Critère optimal = " + str(F))
print("Optimal au point " + str(x))

