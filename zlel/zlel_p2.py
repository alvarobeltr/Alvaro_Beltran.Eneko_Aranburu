#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_p2.py
    :synopsis:

.. moduleauthor:: Eneko Aranburu (earanburu006@gmail.com) eta Alvaro Beltran
(abeltrandenanc002@ikasle.ehu.eus)


"""
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

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
    print("\n========== Nodes voltage to reference ========")
    for i in range(1, n):
        print("e" + str(i) + " = ", "[{:10.9f}]".format(sol[i-1][0]))
    print("\n========== Branches voltage difference ========")
    for i in range(1, b+1):
        print("v" + str(i) + " = ", "[{:10.9f}]".format(sol[i+n-2][0]))
    print("\n=============== Branches currents ==============")
    for i in range(1, b+1):
        print("i" + str(i) + " = ", "[{:10.9f}]".format(sol[i+b+n-2][0]))

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
        header: The header in csv format as string
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
        new_file_path: the filename with the sims_folder_name inserted.
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


def save_as_csv(b, n, filename, sims_folder_name):
    """ This function gnerates a csv file with the name filename.
        First it will save a header and then, it loops and save a line in
        csv format into the file.

    Args:
        | b: # of branches
        | n: # of nodes
        | filename: string with the filename (incluiding the path)
        | sims_folder_name: string with the name of the folder to save the sims
     Returns:
        new_file_path: the filename with the sims_folder_name inserted.

    """
    # Sup .tr
    header = build_csv_header("t", b, n)
    with open(filename, 'w') as file:
        print(header, file=file)
        # Get the indices of the elements corresponding to the sources.
        # The freq parameter cannot be 0 this is why we choose cir_tr[0].
        t = 0
        while t <= 10:
            # for t in tr["start"],tr["end"],tr["step"]
            # Recalculate the Us for the sinusoidal sources

            sol = np.full(2*b+(n-1), t+1, dtype=float)
            # Inserte the time
            sol = np.insert(sol, 0, t)
            # sol to csv
            sol_csv = ','.join(['%.9f' % num for num in sol])
            print(sol_csv, file=file)
            t = round(t + 1, 10)  # 10 decimals to avoid precision errors


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
        filename: string with the name of the file

    Returns:
        An array with:
            - cir_el: np array of strings with the elements to parse. size(1,b)
            - cir_nd: np array with the nodes to the circuit. size(b,4)
            - cir_val: np array with the values of the elements. size(b,3)
            - cir_ctrl: np array of strings with the element which branch
            controls the controlled sources. size(1,b)
            - sim_cmds: np array of strings with the information of the
            simulations.
    Rises:
        SystemExit

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
            PR : Prints information about the circuit
            OP : Prints the operating point of the circuit
            DC : Writes the DC sweep analysis in a cvs file
            TR : Writes the Transient analysis
    Args:
        circuit : Array of 4 elements that describe the circuit
    Returns:
        d : Dictionary with .PR, .OP, .DC and .TR words.
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
    This function expands the matrixes we obtained in the parser in order to
    fit with the number of the branches of the elements added.

    Parameters
    ----------
    circuit: Array formed by the next matrices and another matrix including
    the simulations:
        cir_el: np array of strings with the elements to parse. size(1,b)
        cir_nd: np array with the nodes to the circuit. size(b,4)
        cir_val: np array with the values of the elements. size(b,3)
        cir_ctrl: np array of strings with the element which branch
        controls the controlled sources. size(1,b)

    Returns
    -------
    An array including the next matrices:
        cir_el2: cir_el extended
        cir_nd2: cir_nd extended
        cir_val2: cir_val extended.
        cir_ctrl2: cir_ctrl extended.
    """
    cir_el = circuit[0]
    cir_nd = circuit[1]
    cir_val = circuit[2]
    cir_ctr = circuit[3]
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

    return [np.array(cir_el2, dtype=str), np.array(cir_nd2, dtype=int),
            np.array(cir_val2, dtype=float), np.array(cir_ctr2, dtype=str)]


def getElemPosition(elem, cir_el2):
    """
    Gives the position of an element in cir_el_luz

    Parameters
    ----------
    elem : String with the name of the element
    cir_el_luz : extended np array of strings with the elements to parse.
    size(b,1)

    Returns
    -------
    i : Integer with the position of the element

    """
    for i in range(0, np.size(cir_el2)):
        if cir_el2[i].lower() == elem.lower():
            return i


def getAdarrak(cir_el2):
    """
    This function returns the size of the list of elements obtained from
    cir_el2.

    Parameters
    ----------
    cir_el: np array of strings with the elements to parse. size(1,b)

    Returns
    -------
    b : an integer which represents the number of branches in the circuit.

    """
    return np.size(cir_el2)


def getMNUs(circuit2):
    """
    Gives M, N and Us matrixes thath will be used in Tableau equations:
        M*v + N*i = Us

    Parameters
    ----------
    b : Integer with the number of branches in the circuit
    cir_el_luz : extended np array of strings with the elements to parse.
    size(b,1)
    cir_val_luz : extended np array with the values of the elements.
    size(b,3)
    cir_ctr_luz : extended np array of strings with the element which branch
    controls the controlled sources. size(b,1)

    Returns
    -------
    M : np array that contains the first matrix of Tableau equations. size(b,b)
    N : np array that contains the second matrix of Tableau equations.
    size(b,b)
    Us : np array that contains the third matrix of Tableau equations.

    """
    cir_el2 = circuit2[0]
    cir_val2 = circuit2[2]
    cir_ctr2 = circuit2[3]
    b = getAdarrak(cir_el2)

    M = np.zeros((b, b), dtype=float)
    N = np.zeros((b, b), dtype=float)
    Us = np.zeros((b, 1), dtype=float)
    Bai = True
    for i in range(b):
        print(cir_el2[i][0])
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
            if Bai:
                M[i][i] = 1
                N[i][i] = 1
                Bai = False
            else:
                Bai = True
        elif cir_el2[i, 0][0].lower() == "e":
            j = getElemPosition(cir_ctr2[i], cir_el2)
            M[i][i] = 1
            M[i][j] = cir_val2[i][0]*(-1)
        elif cir_el2[i, 0][0].lower() == "g":
            j = getElemPosition(cir_ctr2[i], cir_el2)
            N[i][i] = 1
            M[i][j] = cir_val2[i][0]*-1
        elif cir_el2[i, 0][0].lower() == "h":
            j = getElemPosition(cir_ctr2[i], cir_el2)
            N[i][i] = 1
            N[i][j] = cir_val2[i][0]*-1
        elif cir_el2[i, 0][0].lower() == "f":
            j = getElemPosition(cir_ctr2[i], cir_el2)
            M[i][i] = 1
            N[i][j] = cir_val2[i][0]*-1
        elif cir_el2[i, 0][0].lower() == "b":
            M[i][i] = 1
            Us[i] = cir_val2[i][0]
        elif cir_el2[i, 0][0].lower() == "y":
            N[i][i] = 1
            Us[i] = cir_val2[i][0]
    # print(f"M={M.shape}, N={N.shape}, Us={np.shape(Us)}")
    return [M, N, Us]


def Tableau(A, M, N, Us):
    """
    Esta función evalúa las ecuaciones del método Tableau,
    utilizando las matrices M, N y Us, junto con la matriz de incidencia
    reducida A.

    Args:
        | **A**: Matriz de incidencia reducida.
        | **M**: Matriz de voltajes.
        | **N**: Matriz de corrientes.
        | **Us**: Vector de elementos no controlados por voltaje.

    Returns:
        | **T**: Matriz Tableau, con todas las ecuaciones en el orden e, v, i.
        | **Sol**: Lista de soluciones de todas las ecuaciones en el mismo
        orden.
    """

    # 🔹 Verificar dimensiones de entrada
    b1, b2 = A.shape
    T_size = b1 + 2 * b2  # Tamaño total de la matriz Tableau

    # 🔹 Inicializar matriz Tableau y vector de soluciones
    T = np.zeros((T_size, T_size), dtype=float)
    u = np.zeros((T_size, 1), dtype=float)

    # 🔹 Transpuesta de A para facilitar cálculos
    A_T = A.T

    # 🔹 Llenar la matriz Tableau con la ecuación de la ley de Kirchhoff
    # de corriente (KCL)
    for i in range(b1):
        for j in range(b2):
            T[i, b1 + b2 + j] = A[i, j]

    # 🔹 Ecuaciones de la ley de Kirchhoff de voltaje (KVL)
    for i in range(b2):
        for j in range(b1):
            T[b1 + i, j] = -A_T[i, j]
        T[b1 + i, b1 + i] = 1  # Elementos de la diagonal principal

    # 🔹 Manejo de M y N con verificación de dimensiones
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

    # 🔹 Convertir Us a un vector 1D para evitar errores
    Us = np.ravel(Us)

    if len(Us) != b2:
        raise ValueError(f"Tamaño de Us incorrecto: {len(Us)}, "
                         f"se esperaba {b2}")

    for i in range(b2):
        u[b1 + b2 + i, 0] = Us[i]

    # 🔹 Verificar si el sistema tiene solución única
    if np.linalg.det(T) == 0:
        raise ValueError("El sistema no tiene solución única: det(T) = 0")

    # 🔹 Resolver el sistema de ecuaciones
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
        filename = "../cirs/all/1_zlel_opamp.cir"

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
    # Verificar qué simulaciones ejecutar
    if op[".OP"]:
        print("Realizar análisis de punto de operación (OP)")
        MNUs = getMNUs(circuit)

        Aa = zl1.getInzidentziaMatrix(n, b, circuit[1])
        A = zl1.getMurriztutakoIntzidentziaMatrix(Aa, n)

        sol = Tableau(A, MNUs[0], MNUs[1], MNUs[2])
        print_solution(sol, b, n)
    if op[".PR"]:
        print("Realizar impresión de información (PR)")
        zl1.print_cir_info(circuit[0], circuit[1], b, n, nodes, el_num)

    if op[".DC"][0]:  # El primer valor indica si se debe hacer la simulación
        start, end, step = op[".DC"][1]
        source = op[".DC"][2]
        print(f"Realizar barrido DC desde {start} hasta {end} con paso {step},"
              f" fuente: {source}")

    if op[".TR"][0]:
        start, end, step = op[".TR"][1]
        print(f"Realizar análisis transitorio desde {start}s hasta {end}s con "
              f"paso {step}s")

    sims_folder_name = "sims"
    filename = save_sim_output(filename, sims_folder_name, ".tr")
    save_as_csv(b, n, filename, sims_folder_name)
    plot_from_cvs(filename, "t", "e1", "Graph1")
