#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. module:: zlel_p3.py
    :synopsis: Put yours

.. moduleauthor:: Put yours

"""
import math
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
        if el[0][0] == "C": ###
            d = True
            pos = ("C", k)
            d_el.append(pos)
        elif el[0][0] == "L": ##
            d = True
            pos = ("L", k)
            d_el.append(pos)
        k += 1
    return [d, d_el]

def Euler_BackWard(A, circuit, elements, h, t, pre_sol):
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
        d_el = d[1]
        M = elements[0]
        N = elements[1]
        u = elements[2]
        v = circuit[2]
        # Ai = zl1.inc_matrix(cir_parser2)
        n = len(A)
        b = len(A[0])
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

def save_as_csv_tr(b, n, filename, MNUs, circuit, start, end, step, operation):
    """ This function generates a csv file with the name filename.
        First it will save a header and then, it loops and save a line in
        csv format into the file making the transient analysis.

    Args:
        | b: # of branches
        | n: # of nodes
        | filename: string with the filename (incluiding the path)
    """

    Aa = zl1.getInzidentziaMatrix(n, b, circuit[1])
    A = zl1.getMurriztutakoIntzidentziaMatrix(Aa, n)
    cir_el = circuit[0]
    cir_val = circuit[2]
    tr = operation[".TR"]
    if not tr[0]:
        sys.exit("No trancient available")
    tr = tr[1]
    header = zl2.build_csv_header("t", b, n)
    filename = zl2.save_sim_output(filename, "sims", ".tr")
    pre_sol = []
    with open(filename, 'w') as file:
        print(header, file=file)
        # Get the indices of the elements corresponding to the sources.
        # The freq parameter cannot be 0 this is why we choose cir_tr[0].
        t = start
        t1=0
        while t <= end:
            for k, i in enumerate(cir_el):
                if (i[0][0] == "B") or (i[0][0] == "Y"):
                    w = 2*math.pi*cir_val[k][1]
                    MNUs[2][k] = cir_val[k][0]*math.sin((w*t) +
                                                   (math.pi*cir_val[k][2]/180))
            Euler_BackWard(circuit, MNUs, tr[2], t1, pre_sol)
            zl3.NR(A, circuit, MNUs)
            sol = zl2.Tableau(A, MNUs[0], MNUs[1], MNUs[2])
            pre_sol = sol
            # Inserte the time
            sol = np.insert(sol, 0, t)
            # sol to csv
            sol_csv = ','.join(['%.9f' % num for num in sol])
            print(sol_csv, file=file)
            t = round(t + step, 10)  # 10 decimals to avoid precision errors
            t1+=1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/3_zlel_arteztailea.cir"
    
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
    zl3.NR(A, circuit, MNUs)
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
        zl3.save_as_csv_dc(b, n, filename, MNUs, circuit, start, step, end, source)

    if op[".TR"][0]:
        start, end, step = op[".TR"][1]
        print(f"Realizar análisis transitorio desde {start}s hasta {end}s con "
              f"paso {step}s")
        save_as_csv_tr(b, n, filename, MNUs, circuit, start, end, step, op)