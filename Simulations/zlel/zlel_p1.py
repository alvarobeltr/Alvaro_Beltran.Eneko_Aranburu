#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_p1.py
    :synopsis: This module contains functions that are used in the main program
    in order to parse the matrixes that will be used to solve the circuit.

.. moduleauthor:: Eneko Aranburu (earanburu006@gmail.com)
    eta Alvaro Beltran (abeltrandenanc002@ikasle.ehu.eus)


"""

import numpy as np
import sys


def cir_parser(filename):
    """
        This function takes a .cir test circuit and parse it into
        4 matrices.
        If the file has not the proper dimensions it warns and exit.

    Args
    ----
    filename: string with the name of the file

    Returns
    -------
    cir_el: np array of strings with the elements to parse. size(1,b)
    cir_nd: np array with the nodes to the circuit. size(b,4)
    cir_val: np array with the values of the elements. size(b,3)
    cir_ctrl: np array of strings with the element which branch
    controls the controlled sources. size(1,b)

    Rises
    -----
    SystemExit

    """
    try:
        cir = np.array(np.loadtxt(filename, dtype=str))
        cir = np.delete(cir, -1, axis=0)
    except ValueError:
        sys.exit("File corrupted: .cir size is incorrect.")
    for x in cir:
        if np.size(x) != 9:
            sys.exit("Sarrerako fitxategiko matrizearen "
                     "neurriak ez dira egokiak.")
    cir_el = np.array(cir[:, 0:1], dtype=str)
    cir_nd = np.array(cir[:, 1:5], dtype=int)
    cir_val = np.array(cir[:, 5:8], dtype=float)
    cir_ctr = np.array(cir[:, 8:9], dtype=str)
    return (cir_el, cir_nd, cir_val, cir_ctr)


def luzatu_cir(cir_el, cir_nd, cir_val, cir_ctr):
    """
    This function expands the matrixes we obtained in the function before
    in order to fit with the number of the branches of the elements added.

    Args
    ----
    cir_el: np array of strings with the elements to parse. size(1,b)
    cir_nd: np array with the nodes to the circuit. size(b,4)
    cir_val: np array with the values of the elements. size(b,3)
    cir_ctrl: np array of strings with the element which branch
    controls the controlled sources. size(1,b)

    Returns
    -------
    cir_el2: np array of strings with the elements to parse.
    size(b,n-1). cir_el extended
    cir_nd2: np array with the nodes to the circuit. size(b,4).
    cir_nd extended
    cir_val2: np array with the values of the elements. size(b,3)
    cir_val extended.
    cir_ctrl2: np array of strings with the element which branch
    controls the controlled sources. size(b,1). cir_ctrl extended.

    """
    cir_el2 = []
    cir_nd2 = []
    cir_val2 = []
    cir_ctr2 = []
    for i in range(0, np.size(cir_el)):
        if cir_el[i][0].lower() == "q":
            cir_el2.append(cir_el[i]+"_be")
            cir_el2.append(cir_el[i]+"_bc")
            cir_nd2.append([cir_nd[i][1], cir_nd[i][2], 0, 0])
            cir_nd2.append([cir_nd[i][1], cir_nd[i][0], 0, 0])
            cir_val2.append(cir_val[i])
            cir_val2.append(cir_val[i])
            cir_ctr2.append(cir_ctr[i])
            cir_ctr2.append(cir_ctr[i])
        elif cir_el[i][0].lower() == "a":
            cir_el2.append(cir_el[i]+"_in")
            cir_el2.append(cir_el[i]+"_ou")
            cir_nd2.append([cir_nd[i][0], cir_nd[i][1], 0, 0])
            cir_nd2.append([cir_nd[i][2], cir_nd[i][3], 0, 0])
            cir_val2.append(cir_val[i])
            cir_val2.append(cir_val[i])
            cir_ctr2.append(cir_ctr[i])
            cir_ctr2.append(cir_ctr[i])

        else:
            cir_el2.append(cir_el[i])
            cir_nd2.append(cir_nd[i])
            cir_val2.append(cir_val[i])
            cir_ctr2.append(cir_ctr[i])
        i += 1

    return [cir_el2, cir_nd2, cir_val2, cir_ctr2]


def getAdarrak(cir_el):
    """
    This function returns the size of the list of elements obtained from
    cir_el.

    Args
    ----
    cir_el: np array of strings with the elements to parse. size(1,b)

    Returns
    -------
    b : an integer which represents the number of branches in the circuit.

    """
    b = 0
    for s in cir_el:
        if s[0].lower() == "a" or s[0].lower() == "q":
            b += 2
        else:
            b += 1
    return b


def getNodes(cir_nd):
    """
    This funcion takes the cir_nd matrix and returns a list made
    up by the nodes of the circuit.

    Args
    ----
    cir_nd: np array with the nodes of the circuit. size(b,4)

    Returns
    -------
    nodes : a sorted-in-ascending-order np array of the nodes

    """
    nodes = np.unique(cir_nd)
    return np.sort(nodes)


def getNodesNumber(cir_nd):
    """
    This funcion returns the size of the list of nodes from cir_nd using
    the getNodes function.

    Args
    ----
    cir_nd: np array with the nodes of the circuit. size(b,4)

    Returns
    -------
    n : an integer which represents the number of nodes in the "nodes" list.

    """
    return np.size(getNodes(cir_nd))


def getEl_num(cir_el):
    """
    This function returns the size of the list of elements obtained from
    cir_el.

    Args
    ----
    cir_el: np array of strings with the elements to parse.
    size(el_num,n-1)

    Returns
    -------
    el_num : an integer which represents the number of elements in the circuit.

    """
    return np.size(cir_el)


def getInzidentziaMatrix(n, b, cir_nd2):
    """
    This function return np array which represents the incident matrix
    of the circuit.

    Args
    ----
    n : an integer which represents the number of nodes in the "nodes" list.
    cir_nd2 : np array with the nodes to the circuit. size(b,4).
    cir_nd extended
    b : an integer which represents the number of branches in the circuit.

    Returns
    -------
    Aa : a np array which represents the incident matrix of the circuit.

    """
    Aa = np.zeros((n, b), dtype=int)  # Crear matriz de ceros de tamaño n x b
    nodes = getNodes(cir_nd2)  # Obtener nodos únicos
    for col, branch in enumerate(cir_nd2):  # Iterar sobre cada rama
        node_start, node_end = branch[0], branch[1]
        if node_start in nodes:
            row_start = np.where(nodes == node_start)[0][0]
            Aa[row_start, col] = 1  # Nodo de inicio
        if node_end in nodes:
            row_end = np.where(nodes == node_end)[0][0]
            Aa[row_end, col] = -1
    return Aa


def getMurriztutakoIntzidentziaMatrix(Aa, n):
    """
    This function return the matrix which will be used to solve the Tableau
    equations.

    Args
    ----
    Aa : a np array which represents the incident matrix of the circuit.
    n : an integer which represents the number of nodes in the "nodes" list.

    Returns
    -------
    A : a np array wich represents the reduced matrix o i_mat.

    """
    return np.array(Aa[1:n, :])


def ErreferentziNodoa(nodes):
    """
    This function is used to detect if the reference-node is missing.

    Args
    ----
    nodes : a sorted-in-ascending-order np array of the nodes.

    Returns
    -------
    None.

    Rises
    -------
    SystemExit.

    """
    if 0 not in nodes:
        sys.exit("Reference node \"0\" is not defined in the circuit.")


def KonexioBakarrekoNodoak(Aa, nodes):
    """
    This function is used to detect if any node is floating.

    Args
    ----
    Aa : a np array which represents the incident matrix of the circuit.
    nodes : a sorted-in-ascending-order np array of the nodes.

    Returns
    -------
    None.

    Rises
    -----
    SystemExit.

    """
    for n, x in enumerate(Aa):
        if np.size(np.flatnonzero(x != 0)) < 2:
            node = nodes[n]
            sys.exit(f"Node {node} is floating")


def TentsioIturriakParaleloan(cir_el2, cir_val2, Aa):
    """
    This function detects if Voltage sources are connected in parallel.

    Args
    ----
    Aa : a np array which represents the incident matrix of the circuit.
    cir_el2: np array of strings with the elements to parse.
    size(b,n-1). cir_el extended
    cir_val2: np array with the values of the elements. size(b,3)
    cir_val extended.

    Returns
    -------
    None.

    Rises
    -------
    SystemExit.

    """
    adarrak = []
    adar_tentsioak = {}

    for adar, x in enumerate(cir_el2):
        if x[0].lower() in {"v", "e", "b"} and x[1] == "_":
            adarrak.append(adar)
            adar_tentsioak[adar] = cir_val2[adar][0]

    for i in range(len(adarrak)):
        for j in range(i + 1, len(adarrak)):
            adar1, adar2 = adarrak[i], adarrak[j]
            if np.array_equal(abs(Aa[:, adar1]), abs(Aa[:, adar2])):
                if adar_tentsioak[adar1] != adar_tentsioak[adar2]:
                    sys.exit(f"Errorea: {cir_el2[adar1]} "
                             "({adar_tentsioak[adar1]}V) eta {cir_el2[adar2]} "
                             "({adar_tentsioak[adar2]}V) tentsio ezberdinekin "
                             "paraleloan daude.")
                if not np.array_equal(Aa[:, adar1], Aa[:, adar2]):
                    sys.exit(f"Errorea: {cir_el2[adar1]} eta {cir_el2[adar2]}"
                             " tentsioberdinekin paraleloan baina norantza "
                             "kontrakoan daude.")


def KorronteIturriakSeriean(cir_el2, cir_nd2, cir_val2, Aa, b):
    """
    This function is used to detect if Current sources are
    connected in serial and they break KCL.

    Args
    ----
    Aa : a np array which represents the incident matrix of the circuit.
    cir_el2: np array of strings with the elements to parse.
    size(b,n-1). cir_el extended
    cir_val2: np array with the values of the elements. size(b,3)
    cir_val extended.
    cir_nd2 : np array with the nodes to the circuit. size(b,4).
    cir_nd extended
    b : an integer which represents the number of branches in the circuit.

    Returns
    -------
    None.

    Rises
    -------
    SystemExit.

    """

    elementos = {}
    nodos_problema = set()
    for indice, elemento in enumerate(cir_el2):
        if elemento[0].lower() in ("i", "g", "y"):
            elementos[indice] = {
                "nodos": cir_nd2[indice],
                "tipo": elemento[0].lower(),
                "valor": cir_val2[indice][0]
                }
    for indice_x, datos_x in elementos.items():
        for indice_y, datos_y in elementos.items():
            if indice_x != indice_y:
                nodos_x = datos_x["nodos"]
                nodos_y = datos_y["nodos"]
                if (nodos_x[0] == nodos_y[0] and nodos_x[1] != nodos_y[1]):
                    nodos_problema.add(nodos_x[0])
                elif (nodos_x[0] == nodos_y[1] and nodos_x[1] != nodos_y[0]):
                    nodos_problema.add(nodos_x[0])
                elif (nodos_x[1] == nodos_y[0] and nodos_x[0] != nodos_y[1]):
                    nodos_problema.add(nodos_x[0])
                elif (nodos_x[1] == nodos_y[1] and nodos_x[0] != nodos_y[0]):
                    nodos_problema.add(nodos_x[1])
    suma_corriente = 0
    for nodo in nodos_problema:
        rama_lista = obtener_ramas(nodo[0], cir_el2, cir_nd2)
        if all(elemento[0].lower() in ("i", "g") for elemento in rama_lista):
            for i in range(b):
                suma_corriente += cir_val2[i][0] * Aa[nodo][i]
            if suma_corriente != 0:
                sys.exit(f"Korrote tentsioak seriean ondorengo nodoan: {nodo}.")


def obtener_ramas(nodo, cir_el2, cir_nd2):
    """
    This function returns a list of the branches connected to a node
    in the circuit.

    Args
    ----
    nodo : an integer which represents a node in the circuit
    cir_el2: np array of strings with the elements to parse.
    size(b,n-1). cir_el extended
    cir_nd2: np array with the nodes to the circuit. size(b,4).
    cir_nd extended

    Returns
    -------
    lista : a list of the branches connected to a node in the circuit.

    """
    lista = []
    for i in range(0, np.size(cir_el2)):
        for x in cir_nd2[i]:
            if x == nodo:
                lista.append(cir_el2[i])
    return lista


def print_cir_info(cir_el, cir_nd, b, n, nodes, el_num):
    """ Prints the info of the circuit:
        |     1.- Elements info
        |     2.- Node info
        |     3.- Branch info
        |     4.- Variable info
    Args
    ----
    cir_el: reshaped cir_el
    cir_nd: reshaped cir_nd. Now it will be a(b,2) matrix
    b: # of branches
    n: # number of nodes
    nodes: an array with the circuit nodes sorted
    el_num:  the # of elements

    """
    # Element info
    print(str(el_num) + ' Elements')
    # Node info
    print(str(n) + ' Different nodes: ' +
          str(nodes))
    # Branch info
    print("\n" + str(b) + " Branches: ")

    for i in range(1, b+1):
        indent = 12  # Number of blanks for indent
        string = ("\t" + str(i) + ". branch:\t" +
                  str(cir_el[i-1][0]) + "i".rjust(indent - len(cir_el[i-1])) +
                  str(i) + "v".rjust(indent - len(str(i))) + str(i) +
                  " = e" + str(cir_nd[i-1, 0]) +
                  " - e" + str(cir_nd[i-1, 1]))
        print(string)

    # Variable info
    print("\n" + str(2*b + (n-1)) + " variables: ")
    # print all the nodes but the first(0 because is sorted)
    for i in nodes[1:]:
        print("e"+str(i)+", ", end="", flush=True)
    for i in range(b):
        print("i"+str(i+1)+", ", end="", flush=True)
    # print all the branches but the last to close it properly
    # It works because the minuimum amount of branches in a circuit must be 2.
    for i in range(b-1):
        print("v"+str(i+1)+", ", end="", flush=True)
    print("v"+str(b))

    # IT IS RECOMMENDED TO USE THIS FUNCTION WITH NO MODIFICATION.


"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
https://stackoverflow.com/questions/19747371/
python-exit-commands-why-so-many-and-when-should-each-be-used
"""
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/examples/0_zlel_V_R_Q.cir"
    # Parse the circuit
    # [cir_el,cir_nd,cir_val,cir_ctr]=cir_parser(filename)
    cir_parser(filename)


#    THIS FUNCTION IS NOT COMPLETE
