#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis:

.. moduleauthor:: Eneko Aranburu (earanburu006@gmail.com) eta Alvaro Beltran (abeltrandenanc002@ikasle.ehu.eus)


"""

import numpy as np
import sys


def cir_parser(filename):
    """
        This function takes a .cir test circuit and parse it into
        4 matices.
        If the file has not the proper dimensions it warns and exit.

    Args:
        filename: string with the name of the file

    Returns:
        | cir_el: np array of strings with the elements to parse. size(1,b)
        | cir_nd: np array with the nodes to the circuit. size(b,4)
        | cir_val: np array with the values of the elements. size(b,3)
        | cir_ctrl: np array of strings with the element which branch
        | controls the controlled sources. size(1,b)

    Rises:
        SystemExit

    """
    try:
        cir = np.array(np.loadtxt(filename, dtype=str))
        cir = np.delete(cir, -1, axis=0)
    except ValueError:
        sys.exit("File corrupted: .cir size is incorrect.")

    # numpy usefull exmaples
    print("================ cir ==========")
    print(cir)
    print("\n======== a = np.array (cir[:,1], dtype = int) ==========")
    a = np.array(cir[:, 1], dtype=int)
    #a = np.array(cir[:, 1:3], dtype=int)
    print(a)
    print("\n======== a = np.append(a,300) ==========")
    a = np.append(a, 300)
    print(a)
    print("\n======== b = a[a > 3] ==========")
    b = a[a > 3]
    print(b)
    print("\n======== c = np.unique(a) ==========")
    c = np.unique(a)
    print(c)
    print("\n======== d = np.flatnonzero(a != 0) ==========")
    d = np.flatnonzero(a != 0)
    print(d)
    print("\n======== e = np.flatnonzero(a == 0) ==========")
    e = np.flatnonzero(a == 0)
    print(e)
    print("\n======== f = np.array(cir[:, 1:2]) ==========")
    f = np.array(cir[:, 1:2])
    print(f)
    print("\n======== g = np.array(cir[2:4, :]) ==========")
    g = np.array(cir[2:4, :])
    print(g)
    print("\n======== h = np.empty([0], dtype=int) ==========")
    h = np.empty([0], dtype=int)
    print(h)
    print("\n======== i = np.append(h, 1) ==========")
    i = np.append(h, 1)
    print(i)
    print("\n======== i[0] = 2 ==========")
    i[0] = 2
    print(i)
    print("\n======== j = np.empty([0], dtype=str ==========")
    j = np.empty([0], dtype=str)
    print(j)
    print("\n======== k = np.append(j, \"123456\") ==========")
    k = np.append(j, "123456")
    print(k)
    print("\n======== k[0] = \"987654321\" ==========")
    k[0] = "987654321"
    print(k)

    ''' https://www.geeksforgeeks.org/modify-numpy-array-to-store-an-arbitrary-length-string/
    The dtype of any numpy array containing string values is the maximum
    length of any string present in the array. Once set, it will only be able
    to store new string having length not more than the maximum length at the
    time of the creation. If we try to reassign some another string value
    having length greater than the maximum length of the existing elements,
    it simply discards all the values beyond the maximum length.'''
    # THIS FUNCTION IS NOT COMPLETE
    cir_el = np.array(cir[:, 0:1], dtype=str)
    cir_nd = np.array(cir[:, 1:5], dtype=int)
    cir_val = np.array(cir[:, 5:8], dtype=float)
    cir_ctr = np.array(cir[:, 8:9], dtype=str)
    return (cir_el, cir_nd, cir_val, cir_ctr)


def luzatu_cir(cir_el, cir_nd, cir_val, cir_ctr):
    '''
    Elementu guztiak zeharkatu ikusteko zeintzuk diren adar bi baino gehiago 
    dituztenak.
    '''
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
    b = 0
    for s in cir_el:
        if s[0].lower() == "a" or s[0].lower() == "q":
            b = b+2
        else:
            b = b+1
    return b


def getNodes(cir_nd):
    nodes = np.unique(cir_nd)
    return np.sort(nodes)


def getNodesNumber(cir_nd):
    return np.size(getNodes(cir_nd))


def getElementNumber(cir_el):
    return np.size(cir_el)


def getInzidentziaMatrix(n, b, cir_nd2):
    Aa = np.zeros((n, b), dtype=int)  # Crear matriz de ceros de tamaño n x b
    nodes = getNodes(cir_nd2)  # Obtener nodos únicos
    
    for col, branch in enumerate(cir_nd2):  # Iterar sobre cada rama
        node_start, node_end = branch[0], branch[1]  # Supongamos que los nodos están en las 2 primeras posiciones
        
        if node_start in nodes:
            row_start = np.where(nodes == node_start)[0][0]  # Encontrar índice de nodo en la lista de nodos
            Aa[row_start, col] = 1  # Nodo de inicio
        
        if node_end in nodes:
            row_end = np.where(nodes == node_end)[0][0]  # Encontrar índice del nodo final
            Aa[row_end, col] = -1  # Nodo de destino    
    return Aa


def getMurriztutakoIntzidentziaMatrix(Aa, n):
    return np.array(Aa[1:n, :])      


def print_cir_info(cir_el, cir_nd, b, n, nodes, el_num):
    """ Prints the info of the circuit:
        |     1.- Elements info
        |     2.- Node info
        |     3.- Branch info
        |     4.- Variable info
    Args:
        | cir_el: reshaped cir_el
        | cir_nd: reshaped cir_nd. Now it will be a(b,2) matrix
        | b: # of branches
        | n: # number of nodes
        | nodes: an array with the circuit nodes sorted
        | el_num:  the # of elements

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
                  str(cir_el[i-1]) + "i".rjust(indent  - len(cir_el[i-1])) +
                  str(i) + "v".rjust(indent  - len(str(i))) + str(i) +
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
