import pandas as pd

data = [100, 102, 104, 200, 202]

series = pd.Series(data, index=["a", "b", "c", "d", "e"])


print(series.loc["a"])
print(series.iloc[2])
print(series[series >= 100])

calories = {"Day1":1750, "Day2":2100, "Day3":1700}

series = pd.Series(calories)

print(series)
series.loc["Day3"] += 300
print(series.loc["Day3"])


print(series[series >= 2000])

