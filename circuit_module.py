"""
defines the rotators and entanglers.
"""
from mindquantum import RX, RY, RZ, U3
from mindquantum import ZZ, XX, YY, U3, Z, X


class CircuitU3CU3():
    def __init__(self):
        self.name = 'U3_CU3'
        self.rotators = [U3]
        self.entanglers = [U3]
        self.params_pre_qubit_pre_rotate_layer = 3
        self.params_pre_qubit_pre_entangle_layer = 3

    def build_rotator_on(self, q, now_pid):
        return self.rotators[0](f'{now_pid:06d}',
                                f'{now_pid+1:06d}',
                                f'{now_pid+2:06d}').on(q), now_pid+3

    def build_entangler_on(self, q1, q2, now_pid):
        return self.entanglers[0](f'{now_pid:06d}',
                                  f'{now_pid+1:06d}',
                                  f'{now_pid+2:06d}').on(q1, q2), now_pid+3
    

class CircuitRYZZ():
    def __init__(self):
        self.name = 'RY_ZZ'
        self.rotators = [RY]
        self.entanglers = [ZZ]
        self.params_pre_qubit_pre_rotate_layer = 1
        self.params_pre_qubit_pre_entangle_layer = 1

    def build_rotator_on(self, q, now_pid):
        return self.rotators[0](f'{now_pid:06d}').on(q), now_pid+1

    def build_entangler_on(self, q1, q2, now_pid):
        return self.entanglers[0](f'{now_pid:06d}').on([q1, q2]), now_pid+1
    

class CircuitRZXX():
    def __init__(self):
        self.name = 'RZ_XX'
        self.rotators = [RZ]
        self.entanglers = [XX]
        self.params_pre_qubit_pre_rotate_layer = 1
        self.params_pre_qubit_pre_entangle_layer = 1

    def build_rotator_on(self, q, now_pid):
        return self.rotators[0](f'{now_pid:06d}').on(q), now_pid+1

    def build_entangler_on(self, q1, q2, now_pid):
        return self.entanglers[0](f'{now_pid:06d}').on([q1, q2]), now_pid+1
    

class CircuitRXRYRZX():
    def __init__(self):
        self.name = 'RXYZ_X'
        self.rotators = [RX, RY, RZ]
        self.entanglers = [X]
        self.params_pre_qubit_pre_rotate_layer = 3
        self.params_pre_qubit_pre_entangle_layer = 0

    def build_rotator_on(self, q, now_pid):
        return self.rotators[0](f'{now_pid:06d}').on(q)+\
            self.rotators[1](f'{now_pid+1:06d}').on(q)+\
            self.rotators[2](f'{now_pid+2:06d}').on(q), now_pid+3

    def build_entangler_on(self, q1, q2, now_pid):
        return self.entanglers[0].on(q1, q2), now_pid


implemented_circuit_modules = {
    'u3cu3': CircuitU3CU3,
    'ryzz': CircuitRYZZ,
    'rzxx': CircuitRZXX,
    'rxyzx': CircuitRXRYRZX,
}
