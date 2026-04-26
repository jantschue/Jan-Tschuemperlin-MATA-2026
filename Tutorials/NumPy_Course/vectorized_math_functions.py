import numpy as np

radii = np.array([1, 2, 3])

print(np.pi * radii ** 2)

array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

print(array1 + array2)
print(array1 - array2)
print(array1 * array2)

scores = np.array([91, 55, 100, 73, 82, 64])
scores[scores < 60] = 0
print(scores)

print(scores == 100)
print(scores >= 60)