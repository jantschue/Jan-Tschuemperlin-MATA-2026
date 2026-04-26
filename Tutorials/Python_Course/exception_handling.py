
try:
    number = int(input("Enter a number: "))
    print(1 / number)

except ZeroDivisionError:
    print("You can't divide by zero")

except ValueError:
    print("Only numbers")

finally:
    print("Do some cleanup here")