#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. module:: zlel_p3.py
    :synopsis: Put yours

.. moduleauthor:: Put yours

"""

import time
import numpy as np
import sys
if __name__ == "zlel.zlel_p4":
    import zlel.zlel_p1 as zl1
    import zlel.zlel_p2 as zl2
    import zlel.zlel_p3 as zl3
else:
    import zlel_p1 as zl1
    import zlel_p2 as zl2
    import zlel_p3 as zl3
    
def dynamic(circuit):
    """
        This function takes a cir_parser2 and returns whether there are
        any dynamic elements. If there are, it also returns a list with the
        position of each one and the type of element.

    Args:
        cir_parser2 : The circuit parser updated

    Returns:
        is_d : True if there is at least one dynamic element, if not False
        nl_el : List with tuples of each dynamic element and its position

    """
    d = False
    d_el = []
    k = 0
    for el in circuit[0]:
        if el[0] == "C":
            d = True
            pos = ("C", k)
            d_el.append(pos)
        elif el[0] == "L":
            d = True
            pos = ("L", k)
            d_el.append(pos)
        k += 1
    return [d, d_el]

def Euler_BackWard(circuit, elements, h, t, pre_sol):
    """

        This function takes a cir_parser2 and elements, and in case there is a
        L or C it changes the elements matrices to calculate Euler Backward.
        If it is on the first iteration of the transient loop it will use the
        starting values and else it uses the values from the previous iteration
        as starting values.

    Args:
        cir_parser2 : The circuit parser updated
        elements : M, N and u matrices
        h : Diference between a time stamp and the next
        t : Iteration number in the transient loop
        pre_sol : Solution of the previous iteration

    """
    d = dynamic(circuit)
    if d[0]:
        d_el = is_d[1]
        M = elements[0]
        N = elements[1]
        u = elements[2]
        v = circuit[2]
        Ai = zl1.inc_matrix(cir_parser2)
        n = len(Ai)
        b = len(Ai[0])
        if t == 0:
            for el, k in d_el:
                if el == "C":
                    M[k][k] = 1
                    N[k][k] = (-1)*h/v[k][0]
                    u[k] = v[k][1]
                else:
                    M[k][k] = (-1)*h/v[k][0]
                    N[k][k] = 1
                    u[k] = v[k][1]
        else:
            for el, k in d_el:
                if el == "C":
                    vc = pre_sol[n-1+k]
                    u[k] = vc
                else:
                    ic = pre_sol[n-1+b+k]
                    u[k] = ic
        elements = [M, N, u]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs_4/all/3_zlel_RC_iragazki.cir"
    cp = zl1.cir_parser(filename)
    op = zl2.operation(cp)
    cp = zl2.cir_parser2(cp)
    Ai = zl1.inc_matrix(cp)
    elements = zl2.elements(cp)
    b = len(cp[0])
    n = len(Ai)
    outf = filename[:-3] + "tr"
    print(dynamic(cp))
    zl2.save_as_csv_tr(b, n, outf, op, elements, cp)