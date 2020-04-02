import numpy as np
from Wolfe_Skel import *
from Visualg import Visualg

def bfgs(oracle, x0, n_iter_max = 5000, eps = 1e-5, visual = False):
    # Listes pour afficher les courbes
    gradient_norm_list = []
    gradient_step_list = []
    loss_list = []
    
    # Initialisation des variables
    x_n = x0
    x_pre = x_n
    W_n = np.eye(x0.size) # Approximation de la hessienne
    loss, gradient = oracle(x_n)
    d_n = -1*np.dot(W_n, gradient) # Direction de descente
    alpha_n, ok = Wolfe(1, x_n, d_n, oracle) 
    x_n = x_pre + alpha_n * d_n
    loss_pre, gradient_pre = loss, gradient
    n = 0

    gradient_norm = np.linalg.norm(gradient)
    while gradient_norm > eps and n < n_iter_max:

        # Calul des nouvelles variables
        loss_n, gradient_n = oracle(x_n)

        # Ecart avec l'itération précédente
        dx = x_n - x_pre 
        dgrad = gradient_n - gradient_pre

        # Calcul de la hessienne
        t_1 = np.outer(dx, dgrad)/np.vdot(dgrad, dx) # Termes impliqués
        t_2 = np.outer(dgrad, dx)/np.vdot(dgrad, dx) # dans le clacul
        t_3 = np.outer(dx, dx)/np.vdot(dgrad, dx)    # de la hessienne
        W_n = np.dot(np.dot(np.eye(x0.size) - t_1, W_n), np.eye(x0.size) - t_2) + t_3

        # Calcul de la direction de descente
        d_n = -1*np.dot(W_n, gradient_n)

        # Calcul du pas de gradient via l'algorithme de Wolfe
        alpha_n, ok = Wolfe(1, x_n, d_n, oracle)

        # Mise à jour des variables
        x_pre = x_n
        x_n = x_pre + alpha_n * d_n
        loss_pre, gradient_pre = loss_n, gradient_n
        gradient_norm = np.linalg.norm(gradient_n)
        
        # Actualisation de l'historique
        gradient_norm_list.append(gradient_norm)
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
