rows = int(input("Rows: "))
columns = int(input("Columns: "))
symbol = input("Enter a symbol to use: ")


for x in range(rows):
    for y in range(columns):
        print(symbol, end="")
    print()
