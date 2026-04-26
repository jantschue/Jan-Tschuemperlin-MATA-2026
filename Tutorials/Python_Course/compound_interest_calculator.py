principal = 0
rate = 0
time = 0

while principal <= 0:
    principal = float(input("Enter the principal amount: "))
    if principal <= 0:
        print("Principle can't be less than or equal to zero")

while rate <= 0:
    rate = float(input("Enter the interest rate: "))
    if rate <= 0:
        print("Interest can't be less than or equal to zero")

while time <= 0:
    time = float(input("Enter the time: "))
    if rate <= 0:
        print("Time can't be less than or equal to zero")

total = principal * (1 + rate/100)**time
print(f"Balance after {time} years: ${total:.2f}")
