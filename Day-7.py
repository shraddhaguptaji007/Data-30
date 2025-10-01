import numpy as np

print("\n\nTask 1\n\n")
matrix = np.arange(9).reshape(3, 3)


print("Matrix:\n", matrix)


print("Shape:", matrix.shape)      
print("Size:", matrix.size)      
print("Dimensions (ndim):", matrix.ndim) 

print("\n\nTask 2\n\n")

data = np.random.normal(loc=0.0, scale=1.0, size=100)


mean = np.mean(data)
std_dev = np.std(data)

print("Mean:", mean)
print("Standard Deviation:", std_dev)

