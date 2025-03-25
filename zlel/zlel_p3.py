#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. module:: zlel_p3.py
    :synopsis: Put yours

.. moduleauthor:: Put yours

"""
import math
import numpy as np
import sys

if __name__ == "zlel.zlel_p3":
    import zlel.zlel_p1 as zl1
    import zlel.zlel_p2 as zl2
else:
    import zlel_p1 as zl1
    import zlel_p2 as zl2

def non_linear(circuit):
    """

        This function takes a circuit and returns whether there are any
        non-linear elements. If there are, it also returns a list with the
        position of each one and the type of element.

    Args:
        circuit : The circuit parser updated

    Returns:
        nl : True if there is at least one non-linear element, if not False
        nl_el : List with tuples of each non-lineal element and its position

    """
    nl = False
    nl_el = []
    k = 0
    for el in circuit[0]:
        if el[0] == "D":
            nl = True
            pos = ("D", k)
            nl_el.append(pos)
        elif (el[0] == "Q") and ("be" in el):
            nl = True
            pos = ("Q", k)
            nl_el.append(pos)
        k += 1
    return [nl, nl_el]
def diode_NR(I0, nD, Vdj):
    """ https://documentation.help/Sphinx/math.html
        Calculates the g and the I of a diode for a NR discrete equivalent
        Given,

        :math:`Id = I_0(e^{(\\frac{V_d}{nV_T})}-1)`

        The NR discrete equivalent will be,

        :math:`i_{j+1} + g v_{j+1} = I`

        where,

        :math:`g = -\\frac{I_0}{nV_T}e^{(\\frac{V_d}{nV_T})}`

        and

        :math:`I = I_0(e^{(\\frac{V_{dj}}{nV_T})}-1) + gV_{dj}`

    Args:
        | I0: Value of I0.
        | nD: Value of nD.
        | Vdj: Value of Vd.

    Return:
        | gd: Conductance of the NR discrete equivalent for the diode.
        | Id: Current independent source of the NR discrete equivalent.

    """

    Vt = 8.6173324e-5*300*nD
    gd = -I0/(nD*Vt)*(math.exp(Vdj/(nD*Vt)))
    I = I0*(math.exp(Vdj/(nD*Vt))-1)
    Id = I + gd*Vdj
    return [gd, Id]
 #   return gd, Id
def Transistor_NR(Ies, Ics, Bf, Vbe, Vbc):
    """

        This function takes the transistors branches voltages, and transistor
        constants and returns Newton Raphson equivalent of the transistor.

    Args:
        Ies : Emitter saturation current
        Ics : Colector saturation current
        Bf : Common emitter current gain
        Vbe : Previous voltage value for BE connection
        Vbc : Previous voltage value for BC connection

    Returns:
        G : Equivalent conductance of the transistor
        Ie : Equivalent current of the transistor on the emitter
        Ic : Equivalent current of the transistor on the collector

    """
    T = 300
    VT = 8.6173324e-5*T
    aF = Bf/(Bf+1)
    aR = Ies/Ics*aF
    g11 = -Ies/VT*math.exp(Vbe/VT)
    g22 = -Ics/VT*math.exp(Vbc/VT)
    g21 = -aF*g11
    g12 = -aR*g22
    Ie = g11*Vbe + g12*Vbc + Ies*(math.exp(Vbe/VT)-1) - aR*Ics*(
        math.exp(Vbc/VT)-1)
    Ic = g21*Vbe + g22*Vbc - aF*Ies*(math.exp(Vbe/VT)-1) + Ics*(
        math.exp(Vbc/VT)-1)
    G = [[g11, g12], [g21, g22]]
    return [G, [Ie, Ic]]

def MNu_D_NR(elements, Diode_NR, k):
    """

        This funcion takes the diode NR values and the elements and returns the
        same elements but with the equivalents replaced in the position given.

    Args:
        elements : Array with M, N and u matrices
        Diode_NR : NR equivalent values g and I
        k : Position of the diode on the parser

    Returns:
        [M, N, u] : The same matrices of elements in the arguments but with
        the NR equivalent replaced in the k position

    """
    M = elements[0]
    N = elements[1]
    u = elements[2]
    g = Diode_NR[0]
    Ij = Diode_NR[1]
    M[k][k] = g
    N[k][k] = 1
    u[k] = Ij
    elements = [M, N, u]

"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
"""
if __name__ == "__main__":
    #  start = time.perf_counter()
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/2_zlel_Q.cir"
        filename = "../cirs/all/2_zlel_1D.cir"


#    end = time.perf_counter()
#    print ("Elapsed time: ")
#    print(end - start) # Time in seconds
