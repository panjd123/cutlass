import numpy as np

A0 = np.loadtxt("tensor_A0.txt")
B0 = np.loadtxt("tensor_B0.txt")
reference_D0 = np.loadtxt("reference_D0.txt")
D0 = np.loadtxt("tensor_D0.txt")
py_reference = np.clip(A0@B0, -8, 7)
print(A0@B0)
print(reference_D0)
print(D0)
# assert np.equal(py_reference, reference_D0).all()
assert np.equal(D0, reference_D0).all()