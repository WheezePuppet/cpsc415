
import numpy as np

PD0 = np.array([.8,0,.2])

PDt = np.array([[.5,.5, 0],
                [.2,.5,.3],
                [ 0,.6,.4]])


print("Probs of D1: {}.".format(PD0.dot(PDt)))
print("Probs of D2: {}.".format(PD0.dot(PDt.dot(PDt))))
print("Probs of D3: {}.".format(PD0.dot(PDt.dot(PDt.dot(PDt)))))
print("Probs of D9999: {}.".format(PD0.dot(np.linalg.matrix_power(PDt,9999))))
stationary = PD0.dot(np.linalg.matrix_power(PDt,9999))

# Get right eigenvectors of PDt.
revecs = np.linalg.eig(PDt.T)[1]

# Get the first one (corresponding to eigenvalue 1).
evec = revecs[:,0]

# Normalize the stationary vector, and compare:
print("{} vs. {}.".format(evec,stationary / np.linalg.norm(stationary)))
print(evec / sum(evec))
