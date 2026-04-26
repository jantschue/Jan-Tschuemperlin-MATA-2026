import pandas as pd

df = pd.read_csv("Pokemon.csv", index_col="Name")

#By column
#print(df[["Name", "HP", "Attack"]].to_string())

#By row
#print(df.loc["Charizard":"Blastoise", ["HP", "Attack"]])
#print(df.iloc[0:11, 0:3])

pokemon = input("Enter a Pokemon name: ")

try:
    print(df.loc[pokemon])

except KeyError:
    print(f"{pokemon} not found")