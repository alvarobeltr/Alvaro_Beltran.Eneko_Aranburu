#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis:

.. moduleauthor:: Eneko Aranburu (earanburu006@gmail.com)
    eta Alvaro Beltran (abeltrandenanc002@ikasle.ehu.eus)


"""

import zlel.zlel_p1 as zl1
import zlel.zlel_p2 as zl2
import zlel.zlel_p3 as zl3
import zlel.zlel_p4 as zl4
import sys

"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
https://stackoverflow.com/questions/19747371/
python-exit-commands-why-so-many-and-when-should-each-be-used
"""
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "cirs/all/0_zlel_node_float.cir"

    cp = zl2.cir_parser(filename)
    circuit = zl2.luzatu_cir(cp)
    pp = zl1.cir_parser(filename)
    nodes = zl1.getNodes(circuit[1])
    zl1.ErreferentziNodoa(nodes)
    b = zl2.getAdarrak(circuit[0])
    n = zl1.getNodesNumber(circuit[1])
    Aa = zl1.getInzidentziaMatrix(n, b, circuit[1])
    A = zl1.getMurriztutakoIntzidentziaMatrix(Aa, n)
    zl1.TentsioIturriakParaleloan(circuit[0], circuit[2], Aa)
    zl1.KorronteIturriakSeriean(circuit[0], circuit[1], circuit[2], Aa, b)
    zl1.KonexioBakarrekoNodoak(Aa, nodes)
    # for i in circuit:
    # print(i)

    op = zl2.getSimulations(cp[4])
    # print(op)

    el_num = zl1.getEl_num(cp[0])
    MNUs = zl2.getMNUs(circuit)
    MNUs = zl4.prepare_dynamic_OP(circuit, MNUs)
    zl3.NR(A, circuit, MNUs)
    # Verificar qué simulaciones ejecutar
    if op[".OP"]:
        sol = zl2.Tableau(A, MNUs[0], MNUs[1], MNUs[2])
        zl2.print_solution(sol, b, n)
    if op[".PR"]:
        zl1.print_cir_info(circuit[0], circuit[1], b, n, nodes, el_num)
        print("\nIncidence Matrix:")
        print(Aa)

    if op[".DC"][0]:  # Indica si se debe hacer la simulación
        start, end, step = op[".DC"][1]
        source = op[".DC"][2]
        zl3.save_as_csv_dc(b, n, filename, MNUs, circuit,
                           start, step, end, source)

    if op[".TR"][0]:
        start, end, step = op[".TR"][1]
        zl4.save_as_csv_tr(b, n, filename, MNUs, circuit, start, end, step, op)
