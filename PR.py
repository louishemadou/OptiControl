import numpy as np
from Wolfe_Skel import Wolfe
from Visualg import Visualg

def Polak_Ribiere(oracle, x0, n_iter_max = 5000, eps = 1e-5, visual = False):
    # Listes pour afficher les courbes
    gradient_norm_list = []
    gradient_step_list = []
    loss_list = []

    # Initialisation des variables
    x_n = x0
    x_pre = x_n
    gradient_pre = oracle(x_pre)[1]

    # Cas n = 0
    gradient_n = gradient_pre
    d_pre = -gradient_pre

    n = 1
    while np.linalg.norm(gradient_n) > eps and n < n_iter_max:
        # Calcul de la direction de descente
        loss_n, gradient_n = oracle(x_n)[:2]
        beta_n = np.dot(gradient_n-gradient_pre, gradient_n)/(np.linalg.norm(gradient_pre)**2)
        d_n = -gradient_n + beta_n * d_pre

        # Calcul du pas optimal (Wolfe)
        alpha_n, ok = Wolfe(1, x_n, d_n, oracle)

        # Actualisation des variables
        x_pre = x_n
        gradient_pre = gradient_n
        d_pre = d_n
        x_n = x_pre + alpha_n * d_n

        # Actualisation de l'historique
        gradient_norm_list.append(np.linalg.norm(gradient_n))
        gradient_step_list.append(alpha_n)
        loss_list.append(loss_n)
        n += 1

    # Résultats
    loss_opt = loss_n
    x_opt = x_n
    gradient_opt = gradient_n

    if visual:
        Visualg(gradient_norm_list, gradient_step_list, loss_list)
    
    return loss_opt, x_opt, gradient_opt
