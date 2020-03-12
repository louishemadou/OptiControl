#!/usr/bin/python

import numpy as np

from numpy import random

#############################################################################
#                                                                           #
#  MONITEUR D'ENCHAINEMENT POUR LE CALCUL DE L'EQUILIBRE D'UN RESEAU D'EAU  #
#                                                                           #
#############################################################################

##### Fonctions fournies dans le cadre du projet

# Donnees du probleme
from Structures_N import A, n, md

# Affichage des resultats
from Visualg import Visualg

# Verification des resultats
from HydrauliqueP import HydrauliqueP
from HydrauliqueD import HydrauliqueD
from Verification import Verification

##### Fonctions a ecrire dans le cadre du projet

# ---> Charger les fonctions associees a l'oracle du probleme,
#      aux algorithmes d'optimisation et de recherche lineaire
#
#      Exemple 1 - le gradient a pas fixe :
#
#                  from OraclePG import OraclePG
#                  from Gradient_F import Gradient_F
#
#      Exemple 2 - le gradient a pas variable :
#
#                  from OraclePG import OraclePG
#                  from Gradient_V import Gradient_V
#                  from Wolfe import Wolfe
#
# ---> A modifier...
# ---> A modifier...
# ---> A modifier...

##### Initialisation de l'algorithme

# ---> La dimension du vecteur dans l'espace primal est n-md
#      et la dimension du vecteur dans l'espace dual est md
#
#      Probleme primal :
#
#                        x0 = 0.1 * random.normal(size=n-md)
#
#      Probleme dual :
#
#                        x0 = 100 + random.normal(size=md)
#
# ---> A modifier...
# ---> A modifier...
# ---> A modifier...

##### Minimisation proprement dite

# ---> Executer la fonction d'optimisation choisie
#
#      Exemple 1 - le gradient a pas fixe :
#
#                  print()
#                  print("ALGORITHME DU GRADIENT A PAS FIXE")
#                  copt, gopt, xopt = Gradient_F(OraclePG, x0)
#
#      Exemple 2 - le gradient a pas variable :
#
#                  print()
#                  print("ALGORITHME DU GRADIENT A PAS VARIABLE")
#                  copt, gopt, xopt = Gradient_V(OraclePG, x0)
#
# ---> A modifier...
# ---> A modifier...
# ---> A modifier...

##### Verification des resultats

# ---> La fonction qui reconstitue les variables hydrauliques
#      du reseau a partir de la solution du probleme s'appelle
#      HydrauliqueP pour le probleme primal, et HydrauliqueD
#      pour le probleme dual
#
#      Probleme primal :
#
#                        qopt, zopt, fopt, popt = HydrauliqueP(xopt)
#
#      Probleme dual :
#
#                        qopt, zopt, fopt, popt = HydrauliqueD(xopt)
#
# ---> A modifier...
# ---> A modifier...
# ---> A modifier...

Verification(A, qopt, zopt, fopt, popt)

