## IonQ, Inc., Copyright (c) 2025,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at iQuHack2025 hosted by MIT and only during the Feb 1-2, 2025
# duration of such event.

import matplotlib.pyplot as plt
from IPython import display

import networkx as nx
import numpy as np
import pandas as pd
import time

from typing import List
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

######################################
from my_graphs import *
from qitevolver import *
from helper import *

##########################################################################
excsolve = True
##########################################################################

# Choose your favorite graph and build your winning ansatz!
graph1 = cycle_graph_c8()
graph2 = complete_bipartite_graph_k88()
graph3 = complete_bipartite_graph_k_nn(5)
graph4 = regular_graph_4_8()
graph5 = cubic_graph_3_16()
graph6 = random_connected_graph_16(p=0.18)
graph7 = expander_graph_n(16)
# graph8 = -> make your own cool graph

graph = graph4
graph

graph_list = [graph1, graph2, graph3, graph4, graph5, graph6, graph7]

for graph_idx, graph in enumerate(graph_list):
    # Visualization will be performed in the cells below;

    ansatz = build_ansatz(graph)
    # ansatz.draw("mpl", fold=-1)

    print(f"Qubits: {ansatz.num_qubits}")

    ham = build_maxcut_hamiltonian(graph)
    ham

    # Set up your QITEvolver and evolve!
    qit_evolver = QITEvolver(ham, ansatz)
    qit_evolver.evolve(
        num_steps=50, lr=0.1, excsolve=excsolve, verbose=True
    )  # lr was 0.5

    shots = 100_000

    # Sample your optimized quantum state using Aer
    backend = AerSimulator()
    optimized_state = ansatz.assign_parameters(qit_evolver.param_vals[-1])
    optimized_state.measure_all()
    counts = backend.run(optimized_state, shots=shots).result().get_counts()

    # Find the sampled bitstring with the largest cut value
    cut_vals = sorted(
        ((bs, compute_cut_size(graph, bs)) for bs in counts), key=lambda t: t[1]
    )
    best_bs = cut_vals[-1][0]

    # Now find the most likely MaxCut solution as sampled from your optimized state
    # We'll leave this part up to you!!!
    most_likely_soln = ""

    print(counts)

    # interpret_solution(graph, best_bs)
    print("Cut value: " + str(compute_cut_size(graph, best_bs)))
    print(graph, best_bs)

    # Brute-force approach with conditional checks

    verbose = False

    G = graph
    n = len(G.nodes())
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = 1.0
    if verbose:
        print(w)

    best_cost_brute = 0
    best_cost_balanced = 0
    best_cost_connected = 0

    verbose = False
    for b in range(2**n):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]

        # Create subgraphs based on the partition
        subgraph0 = G.subgraph([i for i, val in enumerate(x) if val == 0])
        subgraph1 = G.subgraph([i for i, val in enumerate(x) if val == 1])

        bs = "".join(str(i) for i in x)

        # Check if subgraphs are not empty
        if len(subgraph0.nodes) > 0 and len(subgraph1.nodes) > 0:
            cost = 0
            for i in range(n):
                for j in range(n):
                    cost = cost + w[i, j] * x[i] * (1 - x[j])
            if best_cost_brute < cost:
                best_cost_brute = cost
                xbest_brute = x
                XS_brut = []
            if best_cost_brute == cost:
                XS_brut.append(bs)

            outstr = "case = " + str(x) + " cost = " + str(cost)

            if (len(subgraph1.nodes) - len(subgraph0.nodes)) ** 2 <= 1:
                outstr += " balanced"
                if best_cost_balanced < cost:
                    best_cost_balanced = cost
                    xbest_balanced = x
                    XS_balanced = []
                if best_cost_balanced == cost:
                    XS_balanced.append(bs)

            if nx.is_connected(subgraph0) and nx.is_connected(subgraph1):
                outstr += " connected"
                if best_cost_connected < cost:
                    best_cost_connected = cost
                    xbest_connected = x
                    XS_connected = []
                if best_cost_connected == cost:
                    XS_connected.append(bs)
            if verbose:
                print(outstr)

    # And this is how we calculate the shots counted toward scores for each class of the problems

    sum_counts = 0
    for bs in counts:
        if bs in XS_brut:
            sum_counts += counts[bs]

    print(f"Pure max-cut: {sum_counts} out of {shots}")

    sum_balanced_counts = 0
    for bs in counts:
        if bs in XS_balanced:
            sum_balanced_counts += counts[bs]

    print(f"Balanced max-cut: {sum_balanced_counts} out of {shots}")

    sum_connected_counts = 0
    for bs in counts:
        if bs in XS_connected:
            sum_connected_counts += counts[bs]

    print(f"Connected max-cut: {sum_connected_counts} out of {shots}")

    base_score = final_score(
        graph,
        XS_brut,
        counts,
        shots,
        ansatz,
        "base",
        XS_balanced=XS_balanced,
        XS_connected=XS_connected,
    )

    balanced_score = final_score(
        graph,
        XS_brut,
        counts,
        shots,
        ansatz,
        "balanced",
        XS_balanced=XS_balanced,
        XS_connected=XS_connected,
    )

    connected_score = final_score(
        graph,
        XS_brut,
        counts,
        shots,
        ansatz,
        "connected",
        XS_balanced=XS_balanced,
        XS_connected=XS_connected,
    )

    print("Base score: " + str(base_score))
    print("Balanced score: " + str(balanced_score))
    print("Connected score: " + str(connected_score))

    # Visualize your results!
    save_filename = f"conv_graph{graph_idx}_excsolve.png"
    title = f"ExcSolve: {excsolve}\n Graph: {graph_idx}, Scores: Base={base_score}, Balanced={balanced_score}, connected={connected_score}"
    qit_evolver.plot_convergence(save_filename=save_filename, title=title)
