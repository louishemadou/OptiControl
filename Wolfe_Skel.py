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


def Wolfe(alpha, x, D, Oracle):

    # Coefficients de la recherche lineaire

    omega_1 = 0.1
    omega_2 = 0.9

    alpha_min = 0
    alpha_max = np.inf

    ok = 0
    dltx = 0.00000001

    # Algorithme de Fletcher-Lemarechal

    # Appel de l'oracle au point initial
    argout = Oracle(x)
    critere = argout[0]
    gradient = argout[1]

    # Initialisation de l'algorithme
    alpha_n = alpha
    xn = x

    # Boucle de calcul du pas
    # xn represente le point pour la valeur courante du pas,
    # xp represente le point pour la valeur precedente du pas.
    while ok == 0:

        # Point precedent pour tester l'indistinguabilite
        xp = xn

        # Point actuel
        xn = x + alpha_n*D

        # Calcul des conditions de Wolfe
        argout_xn = Oracle(xn)
        critere_xn = argout[0]
        gradient_xn = argout[1]
        if critere_xn > critere + omega_1*alpha_n*np.dot(gradient.T, D):
            alpha_max = alpha_n
            alpha_n = 1/2*(alpha_max + alpha_min)
        else:
            if np.dot(gradient_xn.T, D) < omega_2*np.dot(gradient.T, D):
                alpha_min = alpha_n
                if alpha_max == np.inf:
                    alpha_n = 2*alpha_min
                else:
                    alpha_n = 1/2*(alpha_min + alpha_max)
            else:
                ok = 1

        # Test d'indistinguabilite
        if norm(xn - xp) < dltx:
            ok = 2

    return alpha_n, ok

def Polak_Ribiere(Oracle, x0):
    argout = Oracle(x0)
    critere = argout[0]
    gradient = argout[1]
    eps = 0.000001
    k = 0
    while norm(gradient) > eps:
        if k==1:
            D = - gradient
        else:

