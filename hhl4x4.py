from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.circuit.library import SXGate, QFT
import matplotlib.pyplot as plt
import numpy as np

def CRzz(theta): # Controlled Rzz gate of shape [[exp(i*theta), 0], [0, exp(i*theta)]]
    # 0 - control qbit, 1 - target qbit
    CRzz = QuantumCircuit(2, name=f"CRzz({theta})")
    CRzz.cu(0, 0, theta, 0, 0, 1)
    CRzz.x(1)
    CRzz.cu(0, 0, theta, 0, 0, 1)
    CRzz.x(1)
    return CRzz.to_gate()

def hso(): # Hamiltonian simulation operator
    hso = QuantumCircuit(3, name="exp(iAt_0/16)")
    # 0 - control qbit, 1 - most significant target qbit, 2 - least significant target qbit
    hso.ccz(0, 1, 2)
    hso.crx(0.2, 0, 2).inverse()
    hso.append(SXGate().inverse().control(), [0, 2])
    hso.append(CRzz(0.38), [0, 2])
    hso.crx(0.98, 0, 1).inverse()
    hso.append(CRzz(1.88), [0, 1])
    hso.ccx(0, 1, 2)
    hso.crx(0.59, 0, 1).inverse()
    hso.ccx(0, 1, 2)
    hso.ccz(0, 1, 2)
    return hso.to_gate()


qc = QuantumCircuit(7)
p_a = 0 # position of ancilla qbit
p_C = 1 # position of C register
p_B = 5 # position of B register

# Step 1: state preparation
# Encoding b = 1/2[1, 1, 1, 1]T as 1/2(|00> + |01> + |10> + |11>) in B register
qc.h(p_B)
qc.h(p_B + 1)

# Step 2: QPE (Quantum Phase Estimation)
# Initialization of C register qbits superposition
qc.h(p_C)
qc.h(p_C + 1)
qc.h(p_C + 2)
qc.h(p_C + 3)

# Controlled rotation
n_C = 4 # number of qbits in C register
for i in range(n_C):
    power = 2**i
    qc.append(hso().power(power), [p_C + i, p_B + 1, p_B]) # changing order of b register qbits from MSB to LSB


# IQFT (Inverse Quantum Fourier Transform)
iqft = QFT(n_C).inverse()
qc.append(iqft, [p_C, p_C + 1, p_C + 2, p_C + 3])

# Step 3: ancilla bit rotation
r = 6
for i in range(n_C):
    x = 2**(3-i)
    qc.cry(np.pi*x/(2**(r-1)), p_C + i, p_a)

# Step 4: Uncomputation
# QFT (Quantum Fourier Transform)
qft = QFT(n_C)
qc.append(qft, [p_C, p_C + 1, p_C + 2, p_C + 3])

# Reverse controlled rotation
for i in reversed(range(n_C)):
    power = 2**i
    qc.append(hso().inverse().power(power), [p_C + i, p_B + 1, p_B]) # changing order of b register qbits from MSB to LSB

# Uncomputation of C register qbits superposition
qc.h(p_C)
qc.h(p_C + 1)
qc.h(p_C + 2)
qc.h(p_C + 3)

# Step 5: measurement
# qc.save_statevector()
# qc.measure_all()
a_register = ClassicalRegister(1, name="a")
qc.add_register(a_register)
qc.measure(p_a, a_register)
qc.save_statevector()

# Draw the final circuit
qc.draw(output='mpl')
plt.show()

# Simulate the circuit
simulator = AerSimulator()
qc = transpile(qc, simulator)
result = simulator.run(qc, shots=8192).result()
counts = result.get_counts(qc)
print("Results:", counts)
statevector = result.get_statevector(qc)
state_0000001 = statevector["0000001"]
state_0100001 = statevector["0100001"]
state_1000001 = statevector["1000001"]
state_1100001 = statevector["1100001"]
print("state of 0000001:", state_0000001)
print("state of 0100001:", state_0100001)
print("state of 1000001:", state_1000001)
print("state of 1100001:", state_1100001)
