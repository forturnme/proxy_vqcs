"""
pretrain a model with random gene and save the parameters
supported parameters:
--molfile: the path to molecule file. it is under the folder 'molecule/'. default is 'H4.hdf5'
--epoches: how many epoches to train. default is 100
--batchsize: how many samples in a batch. default is 16
--layers: how many layers in the circuit. default is 10
--circuit: which circuit module to use. default is 'u3cu3'
--lr: learning rate. default is 0.01
the parameter is saved in '{molfile}_{epoches}_{batchsize}_{steps}.pretrain_params.params'
"""
import numpy as np
from evaluator.evaluator import Trainer
from params import Params
from codec import *
from circuit_module import *
from openfermion import MolecularData
import argparse
import time

mol_dir = 'molecule/'

parser = argparse.ArgumentParser()
parser.add_argument('--molfile', type=str, default='H4.hdf5')
parser.add_argument('--epoches', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--layers', type=int, default=10)
parser.add_argument('--circuit', type=str, default='u3cu3')
parser.add_argument('--lr', type=float, default=0.01)
args = parser.parse_args()

# print all parameters
print('molfile:', args.molfile)
print('epoches:', args.epoches)
print('batchsize:', args.batchsize)
print('layers:', args.layers)
print('circuit:', args.circuit)
print('lr:', args.lr)

# load molecule
mol = MolecularData(filename=mol_dir+args.molfile)
mol.load()
n_qubits = mol.n_qubits
n_layers = args.layers
lr = args.lr

# define circuit module
circuit_module = implemented_circuit_modules[args.circuit]()
params = Params(circuit_module, n_qubits, n_layers)
params.random_init()

# define trainer
trainer = Trainer(mol)

# pretrain
for i in range(args.epoches):
    stime = time.time()
    # first sample all discrete codes for a batch
    # then evaluate them and store their gradients
    # finally, update the parameters with lr*(mean(gradients))
    # the optimize purpose is to minimize loss function

    # sample discrete codes
    discrete_codes = []
    for j in range(args.batchsize):
        discrete_codes.append(generate_random_discrete_code(n_qubits, n_layers))
    discrete_codes = np.array(discrete_codes)

    # collect gradients
    gradients = []
    for j in range(args.batchsize):
        # first get params
        ps = params.get_params_from_discrete(discrete_codes[j])
        grads = trainer.get_gradient_discrete(discrete_codes[j], circuit_module, ps)
        # then get gradients, gradients is like [(grads_rotator, grads_entangler), ...]
        gradients.append(params.put_gradients_to_discrete(discrete_codes[j], grads))

    # update parameters
    # first get mean gradients for rotator and entangler
    mean_grads_rotator = np.zeros((n_qubits, n_layers, circuit_module.params_pre_qubit_pre_rotate_layer))
    mean_grads_entangler = np.zeros((n_qubits, n_layers, circuit_module.params_pre_qubit_pre_entangle_layer))
    for j in range(args.batchsize):
        mean_grads_rotator += gradients[j][0]
        mean_grads_entangler += gradients[j][1]
    mean_grads_rotator /= args.batchsize
    mean_grads_entangler /= args.batchsize
    # then update parameters
    params.rotator_params -= lr*mean_grads_rotator
    params.entangler_params -= lr*mean_grads_entangler

    # print how much epochs finished, and estmate how much time left
    print('epoch:', i, 'time:', time.time()-stime, 'left:', (args.epoches-i-1)*(time.time()-stime))

# save parameters
params.save(f'{args.molfile}_{args.epoches}_{args.batchsize}_{args.layers}_{args.circuit}_{args.lr}.pretrain_params.params')
print('pretrain finished.')
