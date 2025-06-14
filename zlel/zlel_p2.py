#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_p2.py

    :synopsis: Utility functions for electric circuit simulation
        (netlist parsing, MNA matrices, CSV export, and visualization)

.. moduleauthor:: Eneko Aranburu (earanburu006@gmail.com) and
     Alvaro Beltran (abeltrandenanc002@ikasle.ehu.eus)

This module provides a set of support functions for simulating electric
circuits defined in .cir (netlist) files. It offers tools for:

- Parsing .cir files and converting them into structured matrix formats.
- Extracting simulation commands (.OP, .PR, .DC, .TR) and interpreting
    their parameters.
- Building the M, N, and Us matrices used in the Modified Nodal Analysis
    (Tableau method).
- Generating and saving simulation results (DC sweep and transient analysis)
    to CSV files.
- Printing formatted simulation results to the console.
- Plotting variables from simulation outputs.
- Expanding composite circuit elements such as transistors and
    controlled sources.
- Solving the complete MNA system and verifying uniqueness of the solution.

This module enables a user to simulate the operating point, DC sweeps,
and time-domain behavior of circuits, and to inspect results
numerically or graphically.

"""

import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import math

if __name__ == "zlel.zlel_p2":
    import zlel.zlel_p1 as zl1
else:
    import zlel_p1 as zl1


def print_solution(sol, b, n):
    """ This function prints the solution with format.

        Args:
            | sol: np array with the solution of the Tableau equations
            | (e_1,..,e_n-1,v_1,..,v_b,i_1,..i_b)
            | b: # of branches
            | n: # of nodes

    """

    # The instructor solution needs to be a numpy array of numpy arrays of
    # float. If it is not, convert it to this format.
    if sol.dtype == np.float64:
        np.set_printoptions(sign=' ')  # Only from numpy 1.14
        tmp = np.zeros([np.size(sol), 1], dtype=float)
        for ind in range(np.size(sol)):
            tmp[ind] = np.array(sol[ind])
        sol = tmp

    tolerance = -1e-9  # Adjust as needed
    print("\n========== Nodes voltage to reference ========")
    for i in range(1, n):
        value = sol[i-1][0]
        print("e" + str(i) + " = ", "[{:10.9f}]".format(
            0.0 if value <= 0 and value >= tolerance else value))
    print("\n========== Branches voltage difference ========")
    for i in range(1, b+1):
        value = sol[i+n-2][0]
        print("v" + str(i) + " = ", "[{:10.9f}]".format(
            0.0 if value <= 0 and value >= tolerance else value))
    print("\n=============== Branches currents ==============")
    for i in range(1, b+1):
        value = sol[i+b+n-2][0]
        print("i" + str(i) + " = ", "[{:10.9f}]".format(
            0.0 if value <= 0 and value >= tolerance else value))
    print("\n================= End solution =================\n")


def build_csv_header(tvi, b, n):
    """ This function build the csv header for the output files.
        First column will be v or i if .dc analysis or t if .tr and it will
        be given by argument tvi.
        The header will be this form,
        t/v/i,e_1,..,e_n-1,v_1,..,v_b,i_1,..i_b

    Args:
        | tvi: "v" or "i" if .dc analysis or "t" if .tran
        | b: # of branches
        | n: # of nodes

    Returns:
        | header: The header in csv format as string
    """
    header = tvi
    for i in range(1, n):
        header += ",e" + str(i)
    for i in range(1, b+1):
        header += ",v" + str(i)
    for i in range(1, b+1):
        header += ",i" + str(i)
    return header


def save_sim_output(filename, sims_folder_name, extension):
    """ This function creates an absolute path to a filename inserting
        a folder given by "sims_folder_name" and changing its extension
        by another given by "extensión" (. needs to be included).

    Args:
        | filename: string with the filename (incluiding the path)
        | sims_folder_name: string with the name of the folder to save the sims
        | extension: new extensión for the file.

    Returns:
        | new_file_path: the filename with the sims_folder_name inserted.

    """

    if not os.path.exists(filename):
        print("file does not exist")
        return
    filename = os.path.abspath(filename)
    dir_path = os.path.dirname(filename)
    base_name, ext = os.path.splitext(os.path.basename(filename))
    new_dir_path = os.path.join(dir_path, sims_folder_name)
    try:
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)
    except OSError as e:
        print(f"Error creating directory: {e}")
        return

    new_filename = f"{base_name}{extension}"
    new_file_path = os.path.join(new_dir_path, new_filename)
    return new_file_path


def save_as_csv_tr(b, n, filename, MNUs, circuit, start, end, step):
    """ This function generates a csv file with the name filename.
        First it will save a header and then, it loops and save a line in
        csv format into the file making the transient analysis.

    Args:
        | b: # of branches
        | n: # of nodes
        | filename: string with the filename (incluiding the path)
        | MNUs : M, N and u matrices
        | circuit : The circuit parser updated
        | start : Start of transient analysis
        | end : End of transient analysis
        | step : Step of transient analysis
    """

    Us = MNUs[2]
    Aa = zl1.getInzidentziaMatrix(n, b, circuit[1])
    A = zl1.getMurriztutakoIntzidentziaMatrix(Aa, n)

    cir_el = circuit[0]
    cir_val = circuit[2]

    header = build_csv_header("t", b, n)
    filename = save_sim_output(filename, "sims", ".tr")
    with open(filename, 'w') as file:
        print(header, file=file)
        # Get the indices of the elements corresponding to the sources.
        # The freq parameter cannot be 0 this is why we choose cir_tr[0].
        t = start
        while t <= end:
            for k, i in enumerate(cir_el):
                if (i[0][0] == "B") or (i[0][0] == "Y"):
                    w = 2*math.pi*cir_val[k][1]
                    Us[k] = cir_val[k][0]*math.sin((w*t) +
                                                   (math.pi*cir_val[k][2]/180))

            sol = Tableau(A, MNUs[0], MNUs[1], Us)
            # Inserte the time
            sol = np.insert(sol, 0, t)
            # sol to csv
            sol_csv = ','.join(['%.9f' % num for num in sol])
            print(sol_csv, file=file)
            t = round(t + step, 10)  # 10 decimals to avoid precision errors


def save_as_csv_dc(b, n, filename, MNUs, circuit, start, step, end, source):
    """ This function gnerates a csv file with the name filename.
        First it will save a header and then, it loops and save a line in
        csv format into the file with the dc solution of the circuit.

    Args:
        | b: # of branches
        | n: # of nodes
        | filename: string with the filename (incluiding the path)
        | MNUs : M, N and u matrices
        | circuit : The circuit parser updated
        | start : Start of DC analysis
        | end : End of DC analysis
        | step : Step of DC analysis
        | source : Name or identifier of the independent source
        | to be swept during the DC analysis.

    """
    if source[0].lower() == "v":
        header = build_csv_header("V", b, n)
    else:
        header = build_csv_header("I", b, n)

    Us = MNUs[2]

    Aa = zl1.getInzidentziaMatrix(n, b, circuit[1])
    A = zl1.getMurriztutakoIntzidentziaMatrix(Aa, n)

    cir_el = circuit[0]
    ext = "_" + source + ".dc"
    filename = save_sim_output(filename, "sims", ext)

    eli = next((k for k, i in enumerate(cir_el) if i[0] == source), None)
    if eli is None:
        raise ValueError(f"Source '{source}' not found in circuit.")

    with open(filename, 'w') as file:
        print(header, file=file)
        v = start
        while v <= end:
            Us[eli] = v
            sol = Tableau(A, MNUs[0], MNUs[1], Us)
            # Insert the time
            sol = np.insert(sol, 0, v)
            # sol to csv
            sol_csv = ','.join(['%.9f' % num for num in sol])
            print(sol_csv, file=file)
            v = v + step


def plot_from_cvs(filename, x, y, title):
    """ This function plots the values corresponding to the x string of the
        file filename in the x-axis and the ones corresponding to the y
        string in the y-axis.
        The x and y strings must mach with some value of the header in the
        csv file filename.

    Args:
        | filename: string with the name of the file (including the path).
        | x: string with some value of the header of the file.
        | y: string with some value of the header of the file.

    """
    data = np.genfromtxt(filename, delimiter=',', skip_header=0,
                         skip_footer=1, names=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(data[x], data[y], color='r', label=title)
    ax1.set_xlabel(x)
    ax1.set_ylabel(y)
    plt.show()


def cir_parser(filename):
    """
        This function takes a .cir test circuit and parse it into
        5 matrices.
        If the file has not the proper dimensions it warns and exit.

    Args:
        | filename: string with the name of the file

    Returns:
        | An array with:
            | - cir_el: np array of strings with the elements to parse.
            | size(1,b)
            | - cir_nd: np array with the nodes to the circuit. size(b,4)
            | - cir_val: np array with the values of the elements. size(b,3)
            | - cir_ctrl: np array of strings with the element which branch
            | controls the controlled sources. size(1,b)
            | - sim_cmds: np array of strings with the information of the
            | simulations.

    Rises:
        | SystemExit

    """
    try:
        cir = np.array(np.loadtxt(filename, dtype=str))
    except ValueError:
        sys.exit("File corrupted: .cir size is incorrect.")

    # Simulazio komandoen lerroak identifikatu (.-z hasten direnak)
    sim_start = np.where(np.char.startswith(cir[:, 0], '.'))[0]

    if sim_start.size == 0:
        sys.exit("Ez da simulazio komando bat ere aurkitu.")

    # Zirkuituaren datuak (simulazio komandoak kenduta)
    cir_data = cir[:sim_start[0], :]

    for x in cir_data:
        if np.size(x) != 9:
            sys.exit("Sarrerako fitxategiko matrizearen neurriak ez "
                     "dira egokiak.")

    cir_el = np.array(cir_data[:, 0:1], dtype=str)
    cir_nd = np.array(cir_data[:, 1:5], dtype=int)
    cir_val = np.array(cir_data[:, 5:8], dtype=float)
    cir_ctr = np.array(cir_data[:, 8:9], dtype=str)

    # Simulazio komandoak
    sim_cmds = cir[sim_start, :]

    return [cir_el, cir_nd, cir_val, cir_ctr, sim_cmds]


def getSimulations(sim_cmds):
    """
    This function takes the sim_cmds matrix and returns a dictionary with
    the different operations. If a operation is shown in the parser, it will
    appear as True in the dictionary and for DC and Transient analysis also
    some related values.
        | PR : Prints information about the circuit
        | OP : Prints the operating point of the circuit
        | DC : Writes the DC sweep analysis in a cvs file
        | TR : Writes the Transient analysis

    Args:
        | circuit : Array of 4 elements that describe the circuit

    Returns:
        | d : Dictionary with .PR, .OP, .DC and .TR words.

    """

    d = {".PR": False, ".OP": False, ".DC": [False, 0], ".TR": [False, 0]}

    for _, cmd in enumerate(sim_cmds):
        cmd_str = cmd[0].lower()

        if cmd_str == '.op':
            d[".OP"] = True

        elif cmd_str == '.tr':
            start = float(cmd[5])
            end = float(cmd[6])
            step = float(cmd[7])
            d[".TR"] = [True, [start, end, step]]

        elif cmd_str == '.pr':
            d[".PR"] = True

        elif cmd_str == '.dc':
            start = float(cmd[5])
            end = float(cmd[6])
            step = float(cmd[7])
            source = cmd[8]
            d[".DC"] = [True, [start, end, step], source]
    return d


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


def getElemPosition(elem, cir_el2):
    """
    Gives the position of an element in cir_el_luz

    Args:
        | elem : String with the name of the element
        | cir_el_luz : extended np array of strings with the elements to parse.
        | size(b,1)

    Returns:
        | i : Integer with the position of the element

    """
    for i in range(0, np.size(cir_el2)):
        if cir_el2[i][0].lower() == elem[0].lower():
            return i


def getAdarrak(cir_el2):
    """
    This function returns the size of the list of elements obtained from
    cir_el2.

    Args:
        | cir_el: np array of strings with the elements to parse. size(1,b)

    Returns:
        | b : an integer which contains the number of branches in the circuit.

    """
    return np.size(cir_el2)


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
    b = getAdarrak(cir_el2)
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
            j = getElemPosition(cir_ctr2[i], cir_el2)
            M[i][i] = 1
            M[i][j] = cir_val2[i][0]*(-1)
        elif cir_el2[i, 0][0].lower() == "g":
            j = getElemPosition(cir_ctr2[i], cir_el2)
            N[i][i] = 1
            M[i][j] = -cir_val2[i][0]
        elif cir_el2[i, 0][0].lower() == "f":
            j = getElemPosition(cir_ctr2[i], cir_el2)
            N[i][i] = 1
            N[i][j] = cir_val2[i][0]*-1
        elif cir_el2[i, 0][0].lower() == "h":
            j = getElemPosition(cir_ctr2[i], cir_el2)
            M[i][i] = 1
            N[i][j] = -cir_val2[i][0]
        elif cir_el2[i, 0][0].lower() == "b":
            M[i][i] = 1
            Us[i] = cir_val2[i][0]
        elif cir_el2[i, 0][0].lower() == "y":
            N[i][i] = 1
            Us[i] = cir_val2[i][0]

    return [M, N, Us]


def Tableau(A, M, N, Us):
    """
    This function evaluates the equations of the Tableau method,
    using the M, N, and Us matrices, along with the reduced
    incidence matrix A.

    Args:
        | A: Reduced incidence matrix.
        | M: Voltage matrix.
        | N: Current matrix.
        | Us: Vector of elements not controlled by voltage.

    Returns:
        | T: Tableau matrix, containing all equations in the order e, v, i.
        | Sol: List of solutions for all equations in the same order.
"""
    b1, b2 = A.shape
    T_size = b1 + 2 * b2

    T = np.zeros((T_size, T_size), dtype=float)
    u = np.zeros((T_size, 1), dtype=float)

    A_T = A.T

    for i in range(b1):
        for j in range(b2):
            T[i, b1 + b2 + j] = A[i, j]

    for i in range(b2):
        for j in range(b1):
            T[b1 + i, j] = -A_T[i, j]
        T[b1 + i, b1 + i] = 1

    if N.shape[0] < b2 or N.shape[1] < b2:
        raise ValueError(f"Dimensiones de N incorrectas: {N.shape}, "
                         f"se esperaba ({b2}, {b2})")

    if M.shape[0] < b2 or M.shape[1] < b2:
        raise ValueError(f"Dimensiones de M incorrectas: {M.shape}, "
                         f"se esperaba ({b2}, {b2})")

    for i in range(b2):
        for j in range(b2):
            T[b1 + b2 + i, b1 + j] = M[i, j]
            T[b1 + b2 + i, b1 + b2 + j] = N[i, j]

    Us = np.ravel(Us)

    if len(Us) != b2:
        raise ValueError(f"Tamaño de Us incorrecto: {len(Us)}, "
                         f"se esperaba {b2}")

    for i in range(b2):
        u[b1 + b2 + i, 0] = Us[i]
    if np.linalg.det(T) == 0:
        sys.exit("Error solving Tableau equations, check if det(T) != 0.")
        # raise ValueError("El sistema no tiene solución única: det(T) = 0")
    sol = np.linalg.solve(T, u)

    return sol


"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
https://stackoverflow.com/questions/19747371/
python-exit-commands-why-so-many-and-when-should-each-be-used
"""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/1_zlel_anpli.cir"

    cp = cir_parser(filename)
    circuit = luzatu_cir(cp)
    for i in circuit:
        print(i)

    op = getSimulations(cp[4])
    print(op)

    b = getAdarrak(circuit[0])
    n = zl1.getNodesNumber(circuit[1])
    nodes = zl1.getNodes(circuit[1])
    el_num = zl1.getEl_num(cp[0])
    MNUs = getMNUs(circuit)
    Aa = zl1.getInzidentziaMatrix(n, b, circuit[1])
    A = zl1.getMurriztutakoIntzidentziaMatrix(Aa, n)
    # Verificar qué simulaciones ejecutar
    if op[".OP"]:
        print("Realizar análisis de punto de operación (OP)")

        sol = Tableau(A, MNUs[0], MNUs[1], MNUs[2])
        print_solution(sol, b, n)
    if op[".PR"]:
        print("Realizar impresión de información (PR)")
        zl1.print_cir_info(circuit[0], circuit[1], b, n, nodes, el_num)
        print("\nIncidence Matrix: ")
        print(Aa)

    if op[".DC"][0]:  # El primer valor indica si se debe hacer la simulación
        start, end, step = op[".DC"][1]
        source = op[".DC"][2]
        print(f"Realizar barrido DC desde {start} hasta {end} con paso {step},"
              f" fuente: {source}")
        save_as_csv_dc(b, n, filename, MNUs, circuit, start, step, end, source)

    if op[".TR"][0]:
        start, end, step = op[".TR"][1]
        print(f"Realizar análisis transitorio desde {start}s hasta {end}s con "
              f"paso {step}s")
        save_as_csv_tr(b, n, filename, MNUs, circuit, start, end, step)
