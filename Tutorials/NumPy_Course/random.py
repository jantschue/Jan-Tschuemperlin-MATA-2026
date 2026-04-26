from random import randint

einses = 0
zweien = 0

for i in range(10000000):
    x = randint(1, 2)
    if x == 1:
        einses += 1
    elif x == 2:
        zweien += 1

print(einses)
print(zweien)