import pandas as pd

df = pd.read_csv("Poke.csv")

#df =df.drop(columns=["Legendary", "No"])

#df = df.dropna(subset=["Type2"])

df = df.fillna({"Type2": "Non"})

df["Type1"] = df["Type1"].replace({"Grass": "GRASS",
                                   "Water": "WATER",
                                   "Fire": "FIRE"})

df["Name"] = df["Name"].str.lower()

df["Legendary"] = df["Legendary"].astype(bool)

df = df.drop_duplicates()

print(df.to_string())