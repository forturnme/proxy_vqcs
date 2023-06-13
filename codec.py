"""
provides 3 translators:
from discrete code to circuit, from continuous code to discrete code and from continuous code to circuit.
the continuous code is an tuple of array with every element in (0, 1), each layer for two element.
the discrete code is an tuple of array with every element as integer in [0, n_qubits)
the first array contains n_qubits*n_depth elements
the second array contains 2*n_depth elements where the first defines where to start entagle, 
and the second defines how long to entangle.
e.g., if the second array gives 2, 3 and n_qubits is 4, then the entagle pattern is 2-3, 3-0, 0-1.
if the first arrat gives 2, 3 and n_qubits is 4, then the rotate gates is on 2, 3, 0, 1.
the continuous code is defined as discrete_code / n_qubits + epsilon.
"""

import numpy as np
from mindquantum import Circuit
from circuit_module import *

epsilon = 1e-3


def continuous_to_discrete(cont, n_qubits):
    """
    translate continuous code to discrete code
    """
    return [np.floor(cont[0] * n_qubits), np.floor(cont[1] * n_qubits)]


def discrete_to_circuit(disc, n_qubits, circuit_type):
    """
    translate discrete code to circuit
    """
    circuit = Circuit()
    now_pid = 0
    n_layers = n_layers = int(len(disc[0].flatten())//2)
    rotate_gene = disc[0].reshape((int(n_layers), 2))
    entangle_gene = disc[1].reshape((int(n_layers), 2))
    for i in range(n_layers):
        for j in range(int(rotate_gene[i][1])+1):
            g, now_pid = circuit_type.build_rotator_on(int((rotate_gene[i][0]+j)%n_qubits), now_pid)
            circuit += g
        for j in range(int(entangle_gene[i][1])):
            g, now_pid = circuit_type.build_entangler_on(int((entangle_gene[i][0]+j)%n_qubits), int((entangle_gene[i][0]+j+1)%n_qubits), now_pid)
            circuit += g
    return circuit


def continuous_to_circuit(cont, n_qubits, circuit_type):
    """
    translate continuous code to circuit
    """
    return discrete_to_circuit(continuous_to_discrete(cont, n_qubits), n_qubits, circuit_type)


def generate_random_continuous_code(n_qubits, n_layers):
    """
    generate random continuous code
    """
    return [(np.random.rand(2, n_layers) + epsilon) / n_qubits,\
            (np.random.rand(2, n_layers) + epsilon) / n_qubits]

def generate_random_discrete_code(n_qubits, n_layers):
    """
    generate random discrete code
    """
    return continuous_to_discrete(generate_random_continuous_code(n_qubits, n_layers), n_qubits)
