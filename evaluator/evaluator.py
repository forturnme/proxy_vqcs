"""
define a class that receives an Quantum circuit and evaluate it
using bfgs from scipy.optimize
the class holds a single molecule and a simulator.
"""
from mindquantum import Circuit, Simulator, Transform, InteractionOperator, FermionOperator, Hamiltonian
from scipy.optimize import minimize
from openfermion import MolecularData
import numpy as np
from codec import *


class Evaluator():
    def __init__(self, mol:MolecularData, use_gpu=False) -> None:
        self.n_qubits = mol.n_qubits
        self.mol_name = mol.name
        self.mol = mol
        self.simulator = Simulator('mqvector' if not use_gpu else 'mqvector_gpu', mol.n_qubits)
        ham_of = mol.get_molecular_hamiltonian()
        inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
        ham_hiq = FermionOperator(inter_ops)
        qubit_ham = Transform(ham_hiq).jordan_wigner()
        self.hamiltonian = Hamiltonian(qubit_ham.compress())

    def evaluate(self, circuit:Circuit, init=None) -> float:
        """
        evaluate the circuit
        """
        n_params = len(circuit.params_name)
        if init is not None:
            pass
        else:
            init = np.pi*0.1*np.random.rand(n_params)
        grad_ops = self.simulator.get_expectation_with_grad(self.hamiltonian, circuit)
        def func(x, grad_ops):
            f, g = grad_ops(x)
            f = np.real(f)[0,0]
            g = np.real(g)[0,0]
            return f, g
        res = minimize(func, init, jac=True, args=(grad_ops), method='BFGS', 
                       options={'disp': True, 'gtol':1e-4})
        return res.fun

    def evaluate_discrete(self, disc, circuit_type):
        """
        evaluate the discrete code
        """
        circuit = discrete_to_circuit(disc, self.n_qubits, circuit_type)
        return self.evaluate(circuit)

    def evaluate_continuous(self, cont, circuit_type):
        """
        evaluate the continuous code
        """
        circuit = continuous_to_circuit(cont, self.n_qubits, circuit_type)
        return self.evaluate(circuit)


class Trainer():
    def __init__(self, mol:MolecularData, use_gpu=False) -> None:
        self.n_qubits = mol.n_qubits
        self.mol_name = mol.name
        self.mol = mol
        self.simulator = Simulator('mqvector' if not use_gpu else 'mqvector_gpu', mol.n_qubits)
        ham_of = mol.get_molecular_hamiltonian()
        inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
        ham_hiq = FermionOperator(inter_ops)
        qubit_ham = Transform(ham_hiq).jordan_wigner()
        self.hamiltonian = Hamiltonian(qubit_ham.compress())

    def get_gradient(self, circuit:Circuit, init) -> float:
        """
        train the circuit for some steps
        """
        grad_ops = self.simulator.get_expectation_with_grad(self.hamiltonian, circuit)
        grad = np.real(grad_ops(init)[1])[0,0]
        return grad
    
    def get_gradient_discrete(self, disc, circuit_type, init):
        """
        train the discrete code for some steps
        """
        circuit = discrete_to_circuit(disc, self.n_qubits, circuit_type)
        return self.get_gradient(circuit, init)
    
    def get_gradient_continuous(self, cont, circuit_type, init):
        """
        train the continuous code for some steps
        """
        circuit = continuous_to_circuit(cont, self.n_qubits, circuit_type)
        return self.get_gradient(circuit, init)
    