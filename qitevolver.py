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


def compute_cut_size(graph, bitstring):
    """
    Get the cut size of the partition of ``graph`` described by the given
    ``bitstring``.
    """
    cut_sz = 0
    for u, v in graph.edges:
        if bitstring[u] != bitstring[v]:
            cut_sz += 1
    return cut_sz


def get_ising_energies(operator: SparsePauliOp, states: np.array):
    """
    Get the energies of the given Ising ``operator`` that correspond to the
    given ``states``.
    """
    # Unroll Hamiltonian data into NumPy arrays
    paulis = np.array([list(ops) for ops, _ in operator.label_iter()]) != "I"
    coeffs = operator.coeffs.real

    # Vectorized energies computation
    energies = (-1) ** (states @ paulis.T) @ coeffs
    return energies


def expected_energy(hamiltonian: SparsePauliOp, measurements: np.array):
    """
    Compute the expected energy of the given ``hamiltonian`` with respect to
    the observed ``measurement``.

    The latter is assumed to by a NumPy records array with fields ``states``
    --describing the observed bit-strings as an integer array-- and ``counts``,
    describing the corresponding observed frequency of each state.
    """

    # energies = get_ising_energies(hamiltonian, measurements["states"])

    # Unroll Hamiltonian data into NumPy arrays
    paulis = np.array([list(ops) for ops, _ in hamiltonian.label_iter()]) != "I"
    coeffs = hamiltonian.coeffs.real

    # Vectorized energies computation
    energies = (-1) ** (measurements["states"] @ paulis.T) @ coeffs

    return np.dot(energies, measurements["counts"]) / measurements["counts"].sum()


class QITEvolver:
    """
    A class to evolve a parametrized quantum state under the action of an Ising
    Hamiltonian according to the variational Quantum Imaginary Time Evolution
    (QITE) principle described in IonQ's latest joint paper with ORNL.
    """

    def __init__(self, hamiltonian: SparsePauliOp, ansatz: QuantumCircuit):
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz

        # Define some constants
        self.backend = AerSimulator()
        self.num_shots = 10000
        self.energies, self.param_vals, self.runtime = list(), list(), list()

    def evolve(
        self,
        num_steps: int,
        lr: float = 0.4,
        excsolve: bool = False,
        verbose: bool = True,
    ):
        """
        Evolve the variational quantum state encoded by ``self.ansatz`` under
        the action of ``self.hamiltonian`` according to varQITE.
        """
        curr_params = np.zeros(self.ansatz.num_parameters)
        for k in range(num_steps):
            # Get circuits and measure on backend
            if excsolve:
                # print("Using ExcSolve!")
                iter_qc = self.get_iteration_circuits_exc(curr_params)
            else:
                iter_qc = self.get_iteration_circuits(curr_params)
            job = self.backend.run(iter_qc, shots=self.num_shots)
            q0 = time.time()
            measurements = job.result().get_counts()
            quantum_exec_time = time.time() - q0

            # Update parameters-- set up defining ODE and step forward
            if excsolve:
                # print("Using ExcSolve!")
                Gmat, dvec, curr_energy = self.get_defining_ode_exc(measurements)
            else:
                Gmat, dvec, curr_energy = self.get_defining_ode(measurements)
            dcurr_params = np.linalg.lstsq(Gmat, dvec, rcond=1e-2)[0]
            curr_params += lr * dcurr_params

            # Progress checkpoint!
            if verbose:
                self.print_status_costum(measurements)
            self.energies.append(curr_energy)
            self.param_vals.append(curr_params.copy())
            self.runtime.append(quantum_exec_time)

    def get_defining_ode(self, measurements: List[dict[str, int]]):
        """
        Construct the dynamics matrix and load vector defining the varQITE
        iteration.
        Here the shifts +-pi/2, 0 are used.
        """
        # Load sampled bitstrings and corresponding frequencies into NumPy arrays
        dtype = np.dtype([("states", int, (self.ansatz.num_qubits,)), ("counts", "f")])
        measurements = [
            np.fromiter(map(lambda kv: (list(kv[0]), kv[1]), res.items()), dtype)
            for res in measurements
        ]
        ## measurements[0] is circuit with no shifts

        # Set up the dynamics matrix by computing the gradient of each Pauli word
        # with respect to each parameter in the ansatz using the parameter-shift rule

        ## Paulis P_\alpha in Eq. (9)
        pauli_terms = [
            SparsePauliOp(op)
            for op, _ in self.hamiltonian.label_iter()
            if set(op) != set("I")
        ]
        ## G from Eq. (7)
        Gmat = np.zeros((len(pauli_terms), self.ansatz.num_parameters))
        for i, pauli_word in enumerate(pauli_terms):  ## iter over rows
            for j, jth_pair in enumerate(  ## iter over columns
                zip(
                    measurements[1::2],  ## +pi/2 shift
                    measurements[2::2],  ## -pi/2 shift
                )
            ):
                ## <P_\alpha> = a_0 + b_0 cos(2\delta) + 2 G sin(2\delta)
                for pm, pm_shift in enumerate(jth_pair):
                    Gmat[i, j] += (-1) ** pm * expected_energy(pauli_word, pm_shift)
                ## Gmat = (2G - (-2G) = 4G from Eq. (9) in https://arxiv.org/abs/2404.16135

        # Set up the load vector
        ## measurements[0] is circuit with no shifts
        curr_energy = expected_energy(self.hamiltonian, measurements[0])
        dvec = np.zeros(len(pauli_terms))
        for i, pauli_word in enumerate(pauli_terms):
            rhs_op_energies = get_ising_energies(pauli_word, measurements[0]["states"])
            rhs_op_energies *= (
                get_ising_energies(self.hamiltonian, measurements[0]["states"])
                - curr_energy
            )
            dvec[i] = (
                -np.dot(rhs_op_energies, measurements[0]["counts"]) / self.num_shots
            )
        return Gmat, dvec, curr_energy

    def get_defining_ode_exc(self, measurements: List[dict[str, int]]):
        """
        Construct the dynamics matrix and load vector defining the varQITE
        iteration but
        using a 2nd order Fourier series coming from generators P^3=P (excitation operators).
        Here the shifts +-pi/4, +-pi/2, 0 are used.
        """
        # Load sampled bitstrings and corresponding frequencies into NumPy arrays
        dtype = np.dtype([("states", int, (self.ansatz.num_qubits,)), ("counts", "f")])
        measurements = [
            np.fromiter(map(lambda kv: (list(kv[0]), kv[1]), res.items()), dtype)
            for res in measurements
        ]
        ## measurements[0] is circuit with no shifts

        # Set up the dynamics matrix by computing the gradient of each Pauli word
        # with respect to each parameter in the ansatz using the parameter-shift rule

        ## Paulis P_\alpha in Eq. (9)
        pauli_terms = [
            SparsePauliOp(op)
            for op, _ in self.hamiltonian.label_iter()
            if set(op) != set("I")
        ]
        ## G from Eq. (7)
        Gmat = np.zeros((len(pauli_terms), self.ansatz.num_parameters))
        for i, pauli_word in enumerate(pauli_terms):  ## iter over rows
            for j, jth_quad in enumerate(  ## iter over columns
                zip(
                    measurements[1::4],  ## +pi/4 shift
                    measurements[2::4],  ## -pi/4 shift
                    measurements[3::4],  ## +pi/2 shift
                    measurements[4::4],  ## -pi/2 shift
                )
            ):
                ## https://arxiv.org/abs/2409.05939 Eq. (25)
                ## <P_\alpha> = a_1 cos(\delta) + a_2 cos(2\delta) + (2G  - b_{11}) sin(2\delta)
                ##              + 1/2 b_{11} sin(2\delta) + c

                ##
                ## +np.pi / 4:   +a1 1/sqrt(2)    0      +b1 1/sqrt(2)       +b2         c
                ## -np.pi / 4:   +a1 1/sqrt(2)    0      -b1 1/sqrt(2)       -b2         c
                ## +np.pi / 2:   0                -a2    +b1                 0           c
                ## -np.pi / 2:   0                -a2    -b1                 0           c

                ## b1 = [f(pi/2) - f(-pi/2)]/2

                ## [f(pi/4) - f(-pi/4)] = sqrt(2) b1 + 2 b2
                ## b2 = 1/2 [f(pi/4) - f(-pi/4) - sqrt(2) b1]
                ## b2 = 1/2 [f(pi/4) - f(-pi/4) - 1/sqrt(2) [f(pi/2) - f(-pi/2)]]

                ## G = b1 + 2 b2
                ##   = [f(pi/2) - f(-pi/2)]/2  +  f(pi/4) - f(-pi/4) - 1/sqrt(2) [f(pi/2) - f(-pi/2)]

                ## Solve system of linear equations to determine a_1, a_2, (2G  - b_{11}), 1/2 b_{11}
                ## shifts are [np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2]
                nrg_vals = [expected_energy(pauli_word, shift) for shift in jth_quad]
                Gmat[i, j] = (
                    +(
                        (nrg_vals[2] - nrg_vals[3]) / 2  # pi/2 - (-pi/2)
                        + nrg_vals[0]  # +pi/4
                        - nrg_vals[1]  # -pi/4
                        - 1 / np.sqrt(2) * (nrg_vals[2] - nrg_vals[3])  # pi/2 - (-pi/2)
                    )
                    * 2
                )  # FIXME  ## Factor 4 due to factor 4 used in original implementation (2G - (-2G) = 4G)

        # Set up the load vector
        ## measurements[0] is circuit with no shifts
        curr_energy = expected_energy(self.hamiltonian, measurements[0])
        dvec = np.zeros(len(pauli_terms))
        for i, pauli_word in enumerate(pauli_terms):
            rhs_op_energies = get_ising_energies(pauli_word, measurements[0]["states"])
            rhs_op_energies *= (
                get_ising_energies(self.hamiltonian, measurements[0]["states"])
                - curr_energy
            )
            dvec[i] = (
                -np.dot(rhs_op_energies, measurements[0]["counts"]) / self.num_shots
            )
        return Gmat, dvec, curr_energy

    def get_iteration_circuits(self, curr_params: np.array):
        """
        Get the bound circuits that need to be evaluated to step forward
        according to QITE.
        Here the shifts +-pi/2, 0 are used.
        """
        # Use this circuit to estimate your Hamiltonian's expected value
        ## First circuit in the list of circuits that will be executed, no shifts are used
        circuits = [self.ansatz.assign_parameters(curr_params)]

        # Use these circuits to compute gradients
        for k in np.arange(curr_params.shape[0]):
            for j in range(2):
                pm_shift = curr_params.copy()
                pm_shift[k] += (-1) ** j * np.pi / 2
                circuits += [self.ansatz.assign_parameters(pm_shift)]
                ## Create list of circuits that will be excuted.
                ## We apply the parameter shifts  (\theta +- pi/2) to each parameter
                ## Each shift is one circuit added to the list
                ## so every second circuit starting from the first is representing a +pi/2 shift
                ## so every second circuit starting from the second is representing a -pi/2 shift
                ## [param1 +pi/2, param1 -pi/2,
                ##  param2 +pi/2, param2 -pi/2,
                ##  param3 +pi/2, param3 -pi/2,
                ##  ...
                ## ]

        # Add measurement gates and return
        [qc.measure_all() for qc in circuits]
        return circuits

    def get_iteration_circuits_exc(self, curr_params: np.array):
        """
        Get the bound circuits that need to be evaluated to step forward
        according to QITE but
        using a 2nd order Fourier series coming from generators P^3=P (excitation operators).
        Here the shifts +-pi/4, +-pi/2, 0 are used.
        """
        # Use this circuit to estimate your Hamiltonian's expected value
        ## First circuit in the list of circuits that will be executed, no shifts are used
        circuits = [self.ansatz.assign_parameters(curr_params)]

        shifts = [np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2]

        # Use these circuits to compute gradients
        for k in np.arange(curr_params.shape[0]):
            for shift in shifts:
                pm_shift = curr_params.copy()
                pm_shift[k] += shift
                circuits += [self.ansatz.assign_parameters(pm_shift)]
                ## Create list of circuits that will be excuted.
                ## We apply the parameter shift rule (\theta +- pi/4, \theta +- pi/2) to each parameter
                ## Each shift is one circuit added to the list
                ## [param1 +pi/4, param1 -pi/4, param1 +pi/2, param1 -pi/2,
                ##  param2 +pi/4, param2 -pi/4, param2 +pi/2, param2 -pi/2,
                ##  param3 +pi/4, param3 -pi/4, param3 +pi/2, param3 -pi/2,
                ##  ...
                ## ]

        # Add measurement gates and return
        [qc.measure_all() for qc in circuits]
        return circuits

    def plot_convergence(self, save_filename: str | None = None, title: str = ""):
        """
        Plot the convergence of the expected value of ``self.hamiltonian`` with
        respect to the (imaginary) time steps.
        """
        plt.plot(self.energies)
        plt.xlabel("(Imaginary) Time step")
        plt.ylabel("Hamiltonian energy")
        plt.title(f"Convergence of the expected energy\n{title}")
        if save_filename is not None:
            plt.savefig(save_filename, bbox_inches="tight", dpi=128)

    def print_status_costum(self, measurements):
        if len(self.energies) == 0:
            return
        print(
            f"step: {len(self.energies)} curr_energy: {self.energies[-1]} num_circuits: {([len(measurements)] * len(self.energies))[-1]} quantum_exec_time: {self.runtime[-1]}s",
            end="\r",
        )

    def print_status(self, measurements):
        """
        Print summary statistics describing a QITE run.
        """
        stats = pd.DataFrame(
            {
                "curr_energy": self.energies,
                "num_circuits": [len(measurements)] * len(self.energies),
                "quantum_exec_time": self.runtime,
            }
        )
        stats.index.name = "step"
        display.clear_output(wait=True)
        display.display(stats)
