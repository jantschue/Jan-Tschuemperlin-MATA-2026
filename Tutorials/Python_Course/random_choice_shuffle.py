import random

options = ("Rock", "Paper", "Scissors")
option = random.choice(options)
cards = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
random.shuffle(cards)


print(option)
print(cards)
