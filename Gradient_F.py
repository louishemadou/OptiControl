#!/usr/bin/python

import numpy as np

from numpy.linalg import norm
from time import process_time

#############################################################################
#                                                                           #
#         RESOLUTION D'UN PROBLEME D'OPTIMISATION SANS CONTRAINTES          #
#                                                                           #
#         Methode du gradient a pas fixe                                    #
#                                                                           #
#############################################################################

from Visualg import Visualg
import oracle

def Gradient_F(Oracle, x0, visual = False):

    # Initialisation des variables

    iter_max = 5000
    gradient_step = 0.0005
    threshold = 0.000001

    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    time_start = process_time()

    x = x0

    # Boucle sur les iterations

    for k in range(iter_max):

        # Valeur du critere et du gradient
        critere, gradient = Oracle(x)

        # Test de convergence
        gradient_norm = norm(gradient)
        if gradient_norm <= threshold:
            break

        # Direction de descente
        direction = -gradient

        # Mise a jour des variables
        x = x + (gradient_step*direction)

        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(gradient_norm)
        gradient_step_list.append(gradient_step)
        critere_list.append(critere)

    # Resultats de l'optimisation

    critere_opt = critere
    gradient_opt = gradient
    x_opt = x
    time_cpu = process_time() - time_start

    print()
    print('Iteration :', k)
    print('Temps CPU :', time_cpu)
    print('Critere optimal :', critere_opt)
    print('Norme du gradient :', norm(gradient_opt))

    # Visualisation de la convergence
    if visual:
        Visualg(gradient_norm_list, gradient_step_list, critere_list)
    return critere_opt, gradient_opt, x_opt
