"""
store and manage parameters.
"""
from joblib import dump, load
import numpy as np
from circuit_module import *
from codec import *

save = dump


class Params():
    def __init__(self, circuit_module, n_qubits, n_layers):
        self.rotator_params = np.zeros((n_qubits, n_layers, circuit_module.params_pre_qubit_pre_rotate_layer))
        self.entangler_params = np.zeros((n_qubits, n_layers, circuit_module.params_pre_qubit_pre_entangle_layer))
        self.circuit_module = circuit_module
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def random_init(self):
        self.rotator_params = np.random.rand(self.n_qubits, self.n_layers, self.circuit_module.params_pre_qubit_pre_rotate_layer) - 0.5
        self.entangler_params = np.random.rand(self.n_qubits, self.n_layers, self.circuit_module.params_pre_qubit_pre_entangle_layer) - 0.5
    
    def save(self, path):
        save(self, path)
    
    def load(self, path):
        params = load(path)
        self.rotator_params = params.rotator_params
        self.entangler_params = params.entangler_params
        self.circuit_module = params.circuit_module
        self.n_qubits = params.n_qubits
        self.n_layers = params.n_layers
        return self
    
    def get_params_from_discrete(self, disc):
        """
        get params with respect to discrete code
        return the param as 1-d array.
        """
        param = []
        n_layers = int(len(disc[0].flatten())//2)
        rotate_gene = disc[0].reshape((int(n_layers), 2))
        entangle_gene = disc[1].reshape((int(n_layers), 2))
        for i in range(n_layers):
            for j in range(int(rotate_gene[i][1])+1):
                for k in range(self.circuit_module.params_pre_qubit_pre_rotate_layer):
                    param.append(self.rotator_params[int((rotate_gene[i][0]+j)%self.n_qubits)][i][k])
            if self.circuit_module.params_pre_qubit_pre_entangle_layer == 0:
                continue
            for j in range(int(entangle_gene[i][1])):
                for k in range(self.circuit_module.params_pre_qubit_pre_entangle_layer):
                    param.append(self.entangler_params[int((entangle_gene[i][0]+j)%self.n_qubits)][i][k])
        return np.array(param)
    
    def set_params_from_discrete(self, disc, param):
        """
        set params with respect to discrete code
        """
        n_layers = int(len(disc[0].flatten())//2)
        rotate_gene = disc[0].reshape(int(n_layers), 2)
        entangle_gene = disc[1].reshape(int(n_layers), 2)
        now_pid = 0
        for i in range(n_layers):
            for j in range(int(rotate_gene[i][1])+1):
                for k in range(self.circuit_module.params_pre_qubit_pre_rotate_layer):
                    self.rotator_params[int((rotate_gene[i][0]+j)%self.n_qubits)][i][k] = param[now_pid]
                    now_pid += 1
            if self.circuit_module.params_pre_qubit_pre_entangle_layer == 0:
                continue
            for j in range(int(entangle_gene[i][1])):
                for k in range(self.circuit_module.params_pre_qubit_pre_entangle_layer):
                    self.entangler_params[int((entangle_gene[i][0]+j)%self.n_qubits)][i][k] = param[now_pid]
                    now_pid += 1

    def put_gradients_to_discrete(self, disc, gradients):
        """
        put gradients to discrete code. the unfilled gradient is treated as 0.
        """
        n_layers = int(len(disc[0].flatten())//2)
        rotate_gene = disc[0].reshape(int(n_layers), 2)
        entangle_gene = disc[1].reshape(int(n_layers), 2)
        rotate_grads = np.zeros((self.n_qubits, n_layers, self.circuit_module.params_pre_qubit_pre_rotate_layer))
        entangle_grads = np.zeros((self.n_qubits, n_layers, self.circuit_module.params_pre_qubit_pre_entangle_layer))
        now_pid = 0
        for i in range(n_layers):
            for j in range(int(rotate_gene[i][1])+1):
                for k in range(self.circuit_module.params_pre_qubit_pre_rotate_layer):
                    rotate_grads[int((rotate_gene[i][0]+j)%self.n_qubits)][i][k] = gradients[now_pid]
                    now_pid += 1
            if self.circuit_module.params_pre_qubit_pre_entangle_layer == 0:
                continue
            for j in range(int(entangle_gene[i][1])):
                for k in range(self.circuit_module.params_pre_qubit_pre_entangle_layer):
                    entangle_grads[int((entangle_gene[i][0]+j)%self.n_qubits)][i][k] = gradients[now_pid]
                    now_pid += 1
        return rotate_grads, entangle_grads
