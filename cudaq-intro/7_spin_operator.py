# Spin operator example

from cudaq import spin

hamiltonian = 0.5*spin.z(0) + spin.x(1) + spin.y(0) + spin.y(0) * spin.y(1)+ spin.x(0)*spin.y(1)*spin.z(2)

# add some more terms
for i in range(2):
  hamiltonian += -2.0*spin.z(i)*spin.z(i+1)

print(hamiltonian)

print('Total number of terms in the spin hamiltonian: ',hamiltonian.get_term_count())