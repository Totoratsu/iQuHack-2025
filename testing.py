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

# Visualization will be performed in the cells below;


def build_ansatz(graph: nx.Graph) -> QuantumCircuit:
    ansatz = QuantumCircuit(graph.number_of_nodes())
    ansatz.h(range(graph.number_of_nodes()))

    theta = ParameterVector(r"$\theta$", graph.number_of_edges())
    for t, (u, v) in zip(theta, graph.edges):
        ansatz.cx(u, v)
        ansatz.ry(t, v)
        ansatz.cx(u, v)

    return ansatz


ansatz = build_ansatz(graph)
ansatz.draw("mpl", fold=-1)

print(f"Qubits: {ansatz.num_qubits}")


def build_maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    """
    Build the MaxCut Hamiltonian for the given graph H = (|E|/2)*I - (1/2)*Σ_{(i,j)∈E}(Z_i Z_j)
    """
    num_qubits = len(graph.nodes)
    edges = list(graph.edges())
    num_edges = len(edges)

    pauli_terms = ["I" * num_qubits]  # start with identity
    coeffs = [-num_edges / 2]

    for u, v in edges:  # for each edge, add -(1/2)*Z_i Z_j
        z_term = ["I"] * num_qubits
        z_term[u] = "Z"
        z_term[v] = "Z"
        pauli_terms.append("".join(z_term))
        coeffs.append(0.5)

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))


ham = build_maxcut_hamiltonian(graph)
ham

##########################################################################
excsolve = False
##########################################################################

# Set up your QITEvolver and evolve!
qit_evolver = QITEvolver(ham, ansatz)
qit_evolver.evolve(num_steps=50, lr=0.4, excsolve=excsolve, verbose=True)  # lr was 0.5

# Visualize your results!
qit_evolver.plot_convergence()


def interpret_solution(graph, bitstring):
    """
    Visualize the given ``bitstring`` as a partition of the given ``graph``.
    """
    pos = nx.spring_layout(graph, seed=42)
    set_0 = [i for i, b in enumerate(bitstring) if b == "0"]
    set_1 = [i for i, b in enumerate(bitstring) if b == "1"]

    plt.figure(figsize=(4, 4))
    nx.draw_networkx_nodes(
        graph, pos=pos, nodelist=set_0, node_color="blue", node_size=700
    )
    nx.draw_networkx_nodes(
        graph, pos=pos, nodelist=set_1, node_color="red", node_size=700
    )

    cut_edges = []
    non_cut_edges = []
    for u, v in graph.edges:
        if bitstring[u] != bitstring[v]:
            cut_edges.append((u, v))
        else:
            non_cut_edges.append((u, v))

    nx.draw_networkx_edges(
        graph, pos=pos, edgelist=non_cut_edges, edge_color="gray", width=2
    )
    nx.draw_networkx_edges(
        graph, pos=pos, edgelist=cut_edges, edge_color="green", width=2, style="dashed"
    )

    nx.draw_networkx_labels(graph, pos=pos, font_color="white", font_weight="bold")
    plt.axis("off")
    plt.show()


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

# This is classical brute force solver results:
# interpret_solution(graph, xbest_brute)
# print(graph, xbest_brute)
# print("\nBest solution = " + str(xbest_brute) + " cost = " + str(best_cost_brute))
# print(XS_brut)

# interpret_solution(graph, xbest_balanced)
# print(graph, xbest_balanced)
# print("\nBest balanced = " + str(xbest_balanced) + " cost = " + str(best_cost_balanced))
# print(XS_balanced)

# interpret_solution(graph, xbest_connected)
# print(graph, xbest_connected)
# print(
#     "\nBest connected = " + str(xbest_connected) + " cost = " + str(best_cost_connected)
# )
# print(XS_connected)
# plt.show()


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


def final_score(graph, XS_brut, counts, shots, ansatz, challenge):

    if challenge == "base":
        sum_counts = 0
        for bs in counts:
            if bs in XS_brut:
                sum_counts += counts[bs]
    elif challenge == "balanced":
        sum_balanced_counts = 0
        for bs in counts:
            if bs in XS_balanced:
                sum_balanced_counts += counts[bs]
        sum_counts = sum_balanced_counts
    elif challenge == "connected":
        sum_connected_counts = 0
        for bs in counts:
            if bs in XS_connected:
                sum_connected_counts += counts[bs]
        sum_counts = sum_connected_counts

    transpiled_ansatz = transpile(ansatz, basis_gates=["cx", "rz", "sx", "x"])
    cx_count = transpiled_ansatz.count_ops()["cx"]
    score = (
        (4 * 2 * graph.number_of_edges())
        / (4 * 2 * graph.number_of_edges() + cx_count)
        * sum_counts
        / shots
    )

    return np.round(score, 5)


print("Base score: " + str(final_score(graph, XS_brut, counts, shots, ansatz, "base")))
print(
    "Balanced score: "
    + str(final_score(graph, XS_brut, counts, shots, ansatz, "balanced"))
)
print(
    "Connected score: "
    + str(final_score(graph, XS_brut, counts, shots, ansatz, "connected"))
)
