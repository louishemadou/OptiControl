#!/usr/bin/python

import numpy as np

from numpy import dot
from numpy.linalg import norm

########################################################################
#                                                                      #
#          RECHERCHE LINEAIRE SUIVANT LES CONDITIONS DE WOLFE          #
#                                                                      #
#          Algorithme de Fletcher-Lemarechal                           #
#                                                                      #
########################################################################

#  Arguments en entree
#
#    alpha  : valeur initiale du pas
#    x      : valeur initiale des variables
#    D      : direction de descente
#    Oracle : nom de la fonction Oracle
#
#  Arguments en sortie
#
#    alphan : valeur du pas apres recherche lineaire
#    ok     : indicateur de reussite de la recherche
#             = 1 : conditions de Wolfe verifiees
#             = 2 : indistinguabilite des iteres

def Wolfe(alpha, x, D, oracle):

    # Coefficients de la recherche lineaire
    omega_1 = 0.1
    omega_2 = 0.9
    alpha_min = 0
    alpha_max = np.inf
    ok = 0
    eps = 0.00000001
    
    loss, gradient = oracle(x)[:2] # Loss et gradient au point initial

    # Initialisation de l'algorithme
    alpha_n = alpha
    x_n = x

    while ok == 0: # Tant qu'on n'a pas trouvé de pas satisfaisant
        # Point précédent pour tester l'indistinguabilité
        x_pre = x_n
        # Point actuel
        x_n = x + alpha_n*D
        loss_n, gradient_n = oracle(x_n)[:2]
        
        if loss_n - loss <= omega_1 * alpha_n * np.dot(gradient, D): # 1ere cond de Wolfe
            if np.dot(gradient_n, D) >= omega_2 * np.dot(gradient, D): # 2eme cond de Wolf
                ok = 1
            else:
                alpha_min = alpha_n
                if alpha_max == np.inf:
                    alpha_n = 2*alpha_min
                else:
                    alpha_n = (1/2) * (alpha_min + alpha_max)
        else:
            alpha_max = alpha_n
            alpha_n = (1/2) * (alpha_min + alpha_max)
        
        # Test d'indistinguabilite
        if np.linalg.norm(x_n - x_pre) < eps:
            ok = 2

    return alpha_n, ok


