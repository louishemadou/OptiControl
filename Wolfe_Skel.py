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
    
    ##### Coefficients de la recherche lineaire

    omega_1 = 0.1
    omega_2 = 0.9

    alpha_min = 0
    alpha_max = np.inf

    ok = 0
    dltx = 0.00000001

    ##### Algorithme de Fletcher-Lemarechal
    
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
        #
        # ---> A completer...
        # ---> A completer...
        
        # Test des conditions de Wolfe
        #
        # - Si les deux conditions de Wolfe sont verifiees,
        #   poser ok = 1 : on sort alors de la boucle while
        # - Sinon, modifier la valeur de alphan et reboucler
        #
        # ---> A completer...
        # ---> A completer...
        
        # Test d'indistinguabilite
        if norm(xn - xp) < dltx:
            ok = 2

    return alpha_n, ok

