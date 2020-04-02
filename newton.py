import numpy as np
from Wolfe_Skel import Wolfe
from Visualg import Visualg

def Newton(oracle, x0, n_iter_max = 5000, eps = 1e-5, visual = False):
    # Listes pour afficher les courbes
    gradient_norm_list = []
    gradient_step_list = []
    loss_list = []

    # Initialisation des variables
    x_n = x0
    n = 0
    gradient_n = oracle(x0)[1]
    while np.linalg.norm(gradient_n) > eps and n < n_iter_max:
        # Calcul de la direction de descente
        loss_n, gradient_n, hessian_n = oracle(x_n)
        d_n = -np.dot(np.linalg.inv(hessian_n), gradient_n)

        # Calcul du pas optimal (Wolfe)
        alpha_n, ok = Wolfe(1, x_n, d_n, oracle)

        # Actualisation des variables
        x_n = x_n + alpha_n * d_n

        # Actualisation de l'historique
        gradient_norm_list.append(np.linalg.norm(gradient_n))
        gradient_step_list.append(alpha_n)
        loss_list.append(loss_n)
        print(np.linalg.norm(gradient_n))
        n += 1

    # Résultats
    loss_opt = loss_n
    x_opt = x_n
    gradient_opt = gradient_n

    if visual:
        Visualg(gradient_norm_list, gradient_step_list, loss_list)
    
    return loss_opt, x_opt, gradient_opt
