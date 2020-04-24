from BFGS import bfgs
from PR import Polak_Ribiere
from newton import Newton
from oracle import *
from Gradient_F import Gradient_F

from time import process_time

def solve(method, visual=False):
    x0 = np.random.normal(size=n - md)
    time_start = process_time()
    if method == "BFGS":
        F, x, G = bfgs(OraclePG, x0, visual=visual)
    if method == "Newton":
        F, x, G = Newton(OraclePH, x0, visual=visual)
    if method == "PR":
        F, x, G = Polak_Ribiere(OraclePG, x0, visual=visual)
    if method == "Gradient_F":
        F, x, G = Gradient_F(OraclePG, x0, visual = visual)
    if method == "Dual":
        u0 = np.random.normal(size=md)
        F, x, G = Newton(OracleDH, u0, visual=visual)
        F, x, G = Gradient_F(OracleDG, u0, visual=visual)
        F, x, G = bfgs(OracleDG, u0, visual = visual)
        F, x, G = Polak_Ribiere(OracleDG, u0, visual=visual)
    cpu_time = process_time() - time_start
    print("Temps d'exécution : ", cpu_time)
    return F, x, G


if __name__ == '__main__':
    F, x, G = solve("Dual", visual=True)
    print("Critère optimal = " + str(F))
    print("Optimal au point " + str(x))
