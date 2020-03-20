import numpy as np
from Wolfe_Skel import *


def bfgs(oracle, x0, iter_max=1000, threshold=1e-8):
    # Initialisation des variables
    x = x0
    x_previous = x
    W = np.eye(x0.size)
    critere, gradient = oracle(x)
    d = - gradient
    alpha, ok = Wolfe(1, x, d, oracle)
    x = x_previous + alpha * d
    gradient_norm = np.linalg.norm(gradient)
    k = 0
    while gradient_norm > threshold and k < iter_max:
        # On actualise le gradient
        gradient_previous = gradient
        critere, gradient = oracle(x)
        # Calcul des écarts
        delta_x = x - x_previous
        delta_grad = gradient - gradient_previous
        # Actualisation de l'approximation de la hessienne
        # Calcul des 3 termes complexes
        t_1 = np.outer(delta_x, delta_grad)/np.vdot(delta_grad, delta_x)
        t_2 = np.outer(delta_grad, delta_x)/np.vdot(delta_grad, delta_x)
        t_3 = np.outer(delta_x, delta_x)/np.vdot(delta_grad, delta_x)
        W = np.dot(np.dot(np.eye(x0.size) - t_1, W),
                   np.eye(x0.size) - t_2) +t_3
        # Calcul de la direction de descente
        d = -1 * np.dot(W, gradient)
        # Calcul du pas de gradient via l'algorithme de Wolfe
        gradient_step, ok = Wolfe(1, x, d, oracle)
        # Mise à jour du point
        x_previous = x
        x = x_previous + gradient_step * d

        gradient_norm = np.linalg.norm(gradient)
        k += 1

    # Resultat de l'optimisation
    critere_opt = critere
    x_opt = x
    gradient_opt = gradient

    return critere_opt, x_opt, gradient_opt
