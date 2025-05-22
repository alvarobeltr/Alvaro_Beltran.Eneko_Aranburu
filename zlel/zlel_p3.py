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
    for k,el in enumerate(circuit[0]):
        if el[0][0] == "D":
            nl = True
            pos = ("D", k)
            nl_el.append(pos)
        elif (el[0][0] == "Q") and ("be" in el):
            nl = True
            pos = ("Q", k)
            nl_el.append(pos)
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
    Ij = I0*(math.exp(Vdj/(nD*Vt))-1)
    Id = Ij + gd*Vdj
    return [gd, Id]


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
    return [M, N, u]


def MNu_Q_NR(elements, Transistor_NR, k):
    """

        This funcion takes the transistor NR equivalent values and the elements
        and returns the same elements but with the equivalents replaced
        starting from the position given.

    Args:
        elements : Array with M, N and u matrices
        Transistor_NR : NR equivalent values G, IE and IC
        k : Starting position of the transistor on the parser

    Returns:
        [M, N, u] : The same matrices of elements in the arguments but with
        the NR equivalent replaced starting on the k position

    """
    M = elements[0]
    N = elements[1]
    u = elements[2]
    G = Transistor_NR[0]
    Ie = Transistor_NR[1][0]
    Ic = Transistor_NR[1][1]
    M[k:k+2, k:k+2] = G
    N[k][k] = 1
    N[k+1][k+1] = 1
    u[k] = Ie
    u[k+1] = Ic
    return [M, N, u]


def NR(A, circuit, elements, e=1e-5, it_max=100):
    """
        This function takes a cir_parser2 and its elements and in case there
        is a D or Q it returns the Newton Raphson equivalent.
    Args:
        circuit : Updated cir_parser
        elements : M, N and u matrices
        e : Error given to solve NR (Default value = 1e-5)
        it_max : Maximum iteration given to solve NR (Default value = 100)

    """

    is_nl = non_linear(circuit)
    nl_el = is_nl[1]
    if is_nl[0]:
        ft = np.full((len(nl_el)), False)
        Vd0 = 0.6
        Vbe0 = 0.6
        Vbc0 = 0.6
        Vs = []
        for el, k in nl_el:
            if el == "D":
                Vs.append(Vd0)
            else:
                Vs.append([Vbe0, Vbc0])
        out = False
        cir_val = circuit[2]
        it = 0
        while (not out) and (it < it_max):
            for i, (el, k) in enumerate(nl_el):
                if el == "D":
                    Vd0 = Vs[i]
                    I0 = cir_val[k][0]
                    n = cir_val[k][1]
                    D_NR = diode_NR(I0, n, Vd0)
                    elements = MNu_D_NR(elements, D_NR, k)
                else:
                    Vbe0 = Vs[i][0]
                    Vbc0 = Vs[i][1]
                    Ies = cir_val[k][0]
                    Ics = cir_val[k][1]
                    BF = cir_val[k][2]
                    Q_NR = Transistor_NR(Ies, Ics, BF, Vbe0, Vbc0)
                    elements = MNu_Q_NR(elements, Q_NR, k)
            sol = zl2.Tableau(A, elements[0], elements[1], elements[2])
            for j, (elm, k) in enumerate(nl_el):
                if elm == "D":
                    VDj = sol[len(A) + k]
                    if (abs(VDj-Vs[j]) < e):
                        ft[j] = True
                        print(ft)
                    Vs[j] = VDj[0]
                else:
                    VBEj = sol[len(A) - 1 + k]
                    VBCj = sol[len(A) + k]
                    if (abs(VBEj-Vs[j][0]) < e) and (abs(VBCj-Vs[j][1]) < e):
                        ft[j] = True
                    Vs[j][0] = VBEj
                    Vs[j][1] = VBCj
            out = np.all(ft)
            it += 1


"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
"""
if __name__ == "__main__":
    #  start = time.perf_counter()
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/2_zlel_2D.cir"
        cp = zl2.cir_parser(filename)
        circuit = zl2.luzatu_cir(cp)
        for i in circuit:
            print(i)

        op = zl2.getSimulations(cp[4])
        print(op)

        b = zl2.getAdarrak(circuit[0])
        n = zl1.getNodesNumber(circuit[1])
        nodes = zl1.getNodes(circuit[1])
        el_num = zl1.getEl_num(cp[0])
        MNUs = zl2.getMNUs(circuit)
        Aa = zl1.getInzidentziaMatrix(n, b, circuit[1])
        A = zl1.getMurriztutakoIntzidentziaMatrix(Aa, n)
        NR(A, circuit, MNUs)
        # Verificar qué simulaciones ejecutar
        if op[".OP"]:
            print("Realizar análisis de punto de operación (OP)")
            sol = zl2.Tableau(A, MNUs[0], MNUs[1], MNUs[2])
            zl2.print_solution(sol, b, n)
        if op[".PR"]:
            print("Realizar impresión de información (PR)")
            zl1.print_cir_info(circuit[0], circuit[1], b, n, nodes, el_num)

        if op[".DC"][0]:  # Indica si se debe hacer la simulación
            start, end, step = op[".DC"][1]
            source = op[".DC"][2]
            print(f"Realizar barrido DC desde {start} hasta {end} "
                  f"con paso {step}, fuente: {source}")
            zl2.save_as_csv_dc(b, n, filename, MNUs, circuit, start, step, end, source)

        if op[".TR"][0]:
            start, end, step = op[".TR"][1]
            print(f"Realizar análisis transitorio desde {start}s hasta {end}s con "
                  f"paso {step}s")
            zl2.save_as_csv_tr(b, n, filename, MNUs, circuit, start, end, step)
