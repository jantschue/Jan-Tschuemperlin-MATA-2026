import csv
file_path = "C:/Users/jants/Desktop/output.csv"

try:
    with open(file_path, "r") as file:
        content = csv.reader(file)
        for line in content:
            print(line)
        print(content)

except FileNotFoundError:
    print("That file was not found")

except PermissionError:
    print("You do not have permission to read that file")