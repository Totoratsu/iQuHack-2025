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


def build_ansatz(graph: nx.Graph) -> QuantumCircuit:
    ansatz = QuantumCircuit(graph.number_of_nodes())
    ansatz.h(range(graph.number_of_nodes()))

    theta = ParameterVector(r"$\theta$", graph.number_of_edges())
    for t, (u, v) in zip(theta, graph.edges):
        ansatz.cx(u, v)
        ansatz.ry(t, v)
        ansatz.cx(u, v)

    return ansatz


def build_ansatz_excitations(graph: nx.Graph) -> QuantumCircuit:

    ansatz = QuantumCircuit(graph.number_of_nodes())
    ansatz.h(range(graph.number_of_nodes()))

    theta = ParameterVector(r"$\theta$", graph.number_of_edges())

    for t, (u, v) in zip(theta, graph.edges):
        ansatz.cx(u, v)
        ansatz.ry(t, v)
        ansatz.cx(u, v)

        ansatz.cx(v, u)
        ansatz.ry(t, u)
        ansatz.cx(v, u)

    return ansatz


# def build_ansatz_excitations(graph: nx.Graph) -> QuantumCircuit:

#     ansatz = QuantumCircuit(graph.number_of_nodes())
#     # ansatz.h(range(graph.number_of_nodes()))

#     ansatz.h(range(graph.number_of_nodes()))

#     ### Do GHZ state
#     # ansatz.h(0)
#     # for i in range(graph.number_of_nodes() - 1):

#     #     ansatz.cx(i, i + 1)
#     ### Flip bits on half of the qubits

#     range_1 = range(int(graph.number_of_nodes() / 2))

#     # for i in range_1:
#     #     ansatz.x(i)

#     theta = ParameterVector(r"$\theta$", graph.number_of_edges())
#     for t, (u, v) in zip(theta, graph.edges):
#         # ansatz.cx(u, v)
#         # ansatz.ry(t, v)
#         # ansatz.cx(u, v)

#         if (u in range_1 and v in range_1) or (u not in range_1 and v not in range_1):
#             continue

#         ### Introduce a single-excitation for every edge instead

#         ansatz.s(u)
#         ansatz.h(v)

#         ansatz.h(u)
#         ansatz.s(v)

#         ansatz.cx(u, v)
#         ansatz.sdg(u)
#         ansatz.h(u)
#         ansatz.tdg(u)
#         ansatz.rz(-t, u)
#         ansatz.h(u)
#         ansatz.s(u)

#         ansatz.t(v)
#         ansatz.rz(t, u)

#         ansatz.cx(u, v)

#         ansatz.h(u)
#         ansatz.sdg(v)

#         ansatz.sdg(u)
#         ansatz.h(v)

#     for t, (u, v) in zip(theta, graph.edges):
#         # ansatz.cx(u, v)
#         # ansatz.ry(t, v)
#         # ansatz.cx(u, v)

#         if (u in range_1 and v in range_1) or (u not in range_1 and v not in range_1):

#             ### Introduce a single-excitation for every edge instead

#             ansatz.s(u)
#             ansatz.h(v)

#             ansatz.h(u)
#             ansatz.s(v)

#             ansatz.cx(u, v)
#             ansatz.sdg(u)
#             ansatz.h(u)
#             ansatz.tdg(u)
#             ansatz.rz(-t, u)
#             ansatz.h(u)
#             ansatz.s(u)

#             ansatz.t(v)
#             ansatz.rz(t, u)

#             ansatz.cx(u, v)

#             ansatz.h(u)
#             ansatz.sdg(v)

#             ansatz.sdg(u)
#             ansatz.h(v)

#     return ansatz


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


def final_score(
    graph, XS_brut, counts, shots, ansatz, challenge, XS_balanced, XS_connected
):

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
