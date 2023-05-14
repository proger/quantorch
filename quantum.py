import torch

def vector_norm(x):
    return torch.linalg.vector_norm(x)

def matrix_norm(x):
    return torch.linalg.matrix_norm(x)

def kron(a, b):
    return torch.kron(a, b)

def dagger(x):
    return x.t().conj()

def ket(bra):
    # |0> = [[1],
    #        [0]]
    # kets are column vectors
    return dagger(bra)

def bra(ket):
    # <0| = [[1, 0]] # bras are row vectors
    return dagger(ket)

def ket_zero():
    return torch.tensor([[1],
                         [0]], dtype=torch.complex64)

def ket_one():
    return torch.tensor([[0],
                         [1]], dtype=torch.complex64)

def ket_plus():
    return torch.tensor([[1],
                         [1]], dtype=torch.complex64) / sq2()

def ket_minus():
    return torch.tensor([[1],
                         [-1]], dtype=torch.complex64) / sq2()

def projector(ket):
    "aka outer product"
    return ket @ bra(ket)

def sq2():
    return torch.sqrt(torch.tensor(2.0, dtype=torch.complex64))

def hadamard():
    return torch.tensor([[1, 1],
                         [1, -1]], dtype=torch.complex64) / sq2()

def N():
    "not"
    return projector(ket_plus()) - projector(ket_minus())

def P():
    "phase swap"
    return projector(ket_zero()) - projector(ket_one())

"""
x,y,z gates are related to the Bloch sphere
"""

def pauli_x():
    "also not"
    return projector(ket_plus()) - projector(ket_minus()) # defined via eigendecomposition

def pauli_y():
    return torch.tensor([[0, -1j],
                         [1j, 0]], dtype=torch.complex64)

def pauli_z():
    "also phase swap"
    return projector(ket_zero()) - projector(ket_one())

assert torch.allclose(pauli_x(), hadamard() @ pauli_z() @ hadamard())
assert torch.allclose(pauli_z(), hadamard() @ pauli_x() @ hadamard())

def cnot():
    return torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=torch.complex64)


def swap():
    return torch.tensor([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=torch.complex64)

def epr_phi_plus():
    return torch.tensor([[1],
                         [0],
                         [0],
                         [1]], dtype=torch.complex64) / sq2()

#assert epr_phi_plus().norm() == 1 # not proper torch

# def repeater():
#     return torch.tensor([[1, 0, 0, 0],

assert torch.allclose(epr_phi_plus(), cnot() @ kron(ket_plus(), ket_zero()))
assert torch.allclose(epr_phi_plus(), (kron(ket_zero(), ket_zero()) + kron(ket_one(), ket_one())) / sq2())


# bra(o) A ket(o) = Expectation value of A

# expectation of the projector: probability of a state



# P(s<10|psi) = <psi| P<10 |psi>