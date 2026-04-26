import numpy as np
print(np.__version__)

wuerfe = np.random.randint(1, 3, size=1_000_000_000)  # 1 oder 2, 10 Mio. mal

einses = np.sum(wuerfe == 1)
zweien = np.sum(wuerfe == 2)

print(einses)
print(zweien)