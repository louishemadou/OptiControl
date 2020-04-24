from BFGS import bfgs
from PR import Polak_Ribiere
from newton import Newton
from oracle import *
from Gradient_F import Gradient_F

from time import process_time

def solve(x0, method, oracle, visual=False):
    time_start = process_time()
    if method == "BFGS":
        F, x, G = bfgs(oracle, x0, visual=visual)
    if method == "Newton":
        F, x, G = Newton(oracle, x0, visual=visual)
    if method == "PR":
        F, x, G = Polak_Ribiere(oracle, x0, visual=visual)
    if method == "Gradient_F":
        F, x, G = Gradient_F(oracle, x0, visual = visual)
    cpu_time = process_time() - time_start
    print("Temps d'exécution : ", cpu_time)
    return F, x, G


if __name__ == '__main__':
    x0_P = np.random.normal(size = n - md)
    x0_D = np.random.normal(size = md)
    F, x, G = solve(x0_D, "BFGS", OracleDG, visual=True)
    print("Critère optimal = " + str(F))
    print("Optimal au point " + str(x))
