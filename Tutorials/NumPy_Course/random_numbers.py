import numpy as np

rng = np.random.default_rng()

print(rng.integers(low=1, high=101, size=(3, 2)))

np.random.seed(seed=1)
print(np.random.uniform(low=-1, high=1, size=(3, 2)))

array = np.array([1, 2, 3, 4, 5])
rng.shuffle(array)
print(array)

fruits = np.array(["apple", "orange", "banana", "coconut", "pineaplle"])
fruits = rng.choice(fruits, size = (3, 3))
print(fruits)