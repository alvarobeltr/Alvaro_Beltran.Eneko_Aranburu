# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 13:01:51 2025

@author:
"""

import numpy as np

if __name__ == "zlel.zlel_p5":
    import zlel.zlel_p1 as zl1
else:
    import zlel_p1 as zl1


def luzatu_cir(circuit):
    """
    Expands circuit matrices to handle multi-branch elements like transistors
    and controlled sources.

    Args:
        | circuit: list of np.arrays
            | - cir_el: np array of strings with the elements to parse.
            | size(1,b)
            | - cir_nd: np array with the nodes to the circuit. size(b,4)
            | - cir_val: np array with the values of the elements. size(b,3)
            | - cir_ctrl: np array of strings with the element which branch
            | controls the controlled sources. size(1,b)

    Returns:
        | Tuple of np.arrays:
            | - cir_el2: expanded element names
            | - cir_nd2: expanded node definitions
            | - cir_val2: expanded values
            | - cir_ctrl2: expanded controls
    """
    cir_el, cir_nd, cir_val, cir_ctr, _ = circuit

    cir_el2, cir_nd2, cir_val2, cir_ctr2 = [], [], [], []

    for i in range(np.size(cir_el)):
        element = cir_el[i][0]

        if element[0].lower() == "q":
            # Expand transistor into two pseudo-branches
            cir_el2 += [[element + "_be"], [element + "_bc"]]
            cir_nd2 += [[cir_nd[i][1], cir_nd[i][2], 0, 0],
                        [cir_nd[i][1], cir_nd[i][0], 0, 0]]
            cir_val2 += [cir_val[i], cir_val[i]]
            cir_ctr2 += [cir_ctr[i], cir_ctr[i]]

        # elif element[0].lower() == "k":
        #     cir_el2 += [[element + "_be"], [element + "_ce"]]
        #     cir_nd2 += [[cir_nd[i][1], cir_nd[i][2], 0, 0],
        #                 [cir_nd[i][0], cir_nd[i][2], 0, 0]]
        #     cir_val2 += [cir_val[i], cir_val[i]]
        #     cir_ctr2 += [cir_ctr[i], cir_ctr[i]]

        elif element[0].lower() == "a":
            # Expand controlled source into input/output components
            cir_el2 += [[element + "_in"], [element + "_ou"]]
            cir_nd2 += [[cir_nd[i][0], cir_nd[i][1], 0, 0],
                        [cir_nd[i][2], cir_nd[i][3], 0, 0]]
            cir_val2 += [cir_val[i], cir_val[i]]
            cir_ctr2 += [cir_ctr[i], cir_ctr[i]]

        else:
            # Keep standard elements unchanged
            cir_el2.append(cir_el[i])
            cir_nd2.append(cir_nd[i])
            cir_val2.append(cir_val[i])
            cir_ctr2.append(cir_ctr[i])

    cir_el2 = np.reshape(np.array(cir_el2), (-1, 1))
    cir_nd2 = np.reshape(np.array(cir_nd2), (-1, 4))
    cir_val2 = np.reshape(np.array(cir_val2), (-1, 3))
    cir_ctr2 = np.reshape(np.array(cir_ctr2), (-1, 1))

    return [cir_el2, cir_nd2, cir_val2, cir_ctr2]


def getMNUs(circuit2):
    """
    Gives M, N and Us matrixes thath will be used in Tableau equations:
        M*v + N*i = Us

    Args:
        | b : Integer with the number of branches in the circuit
        | cir_el_luz : extended np array of strings with the elements to parse.
        | size(b,1)
        | cir_val_luz : extended np array with the values of the elements.
        | size(b,3)
        | cir_ctr_luz : extended np array of strings with the element which
        | branch controls the controlled sources. size(b,1)

    Returns:
        | M : np array that contains the first matrix of Tableau equations.
        | N : np array that contains the second matrix of Tableau equations.
        | size(b,b)
        | Us : np array that contains the third matrix of Tableau equations.
"""

    cir_el2 = circuit2[0]
    cir_val2 = circuit2[2]
    cir_ctr2 = circuit2[3]
    b = zl1.getAdarrak(cir_el2)
    M = np.zeros((b, b), dtype=float)
    N = np.zeros((b, b), dtype=float)
    Us = np.zeros((b, 1), dtype=float)
    for i in range(b):
        if cir_el2[i, 0][0].lower() == "r":
            M[i][i] = 1
            N[i][i] = -cir_val2[i][0]
        elif cir_el2[i, 0][0].lower() == "v":
            M[i][i] = 1
            Us[i] = cir_val2[i][0]
        elif cir_el2[i, 0][0].lower() == "i":
            N[i][i] = 1
            Us[i] = cir_val2[i][0]
        elif cir_el2[i, 0][0].lower() == "a":
            if "ou" in cir_el2[i, 0].lower():
                M[i][i-1] = 1
            else:
                N[i][i] = 1
        elif cir_el2[i, 0][0].lower() == "e":
            j = zl1.getElemPosition(cir_ctr2[i], cir_el2)
            M[i][i] = 1
            M[i][j] = cir_val2[i][0]*(-1)
        elif cir_el2[i, 0][0].lower() == "g":
            j = zl1.getElemPosition(cir_ctr2[i], cir_el2)
            N[i][i] = 1
            M[i][j] = -cir_val2[i][0]
        elif cir_el2[i, 0][0].lower() == "f":
            j = zl1.getElemPosition(cir_ctr2[i], cir_el2)
            N[i][i] = 1
            N[i][j] = cir_val2[i][0]*-1
        elif cir_el2[i, 0][0].lower() == "h":
            j = zl1.getElemPosition(cir_ctr2[i], cir_el2)
            M[i][i] = 1
            N[i][j] = -cir_val2[i][0]
        elif cir_el2[i, 0][0].lower() == "b":
            M[i][i] = 1
            Us[i] = cir_val2[i][0]
        elif cir_el2[i, 0][0].lower() == "y":
            N[i][i] = 1
            Us[i] = cir_val2[i][0]
        # elif cir_el2[i, 0][0].lower() == "k":
        #     if "be" in cir_el2[i, 0].lower():
        #         M[i][i] = 1
        #         N[i][i] = -cir_val2[i][1]
        #         Us[i] = cir_val2[i][0]
        #     else:
        #         N[i][i] = 1
        #         N[i][i-1] = -cir_val2[i][2]

    return [M, N, Us]
