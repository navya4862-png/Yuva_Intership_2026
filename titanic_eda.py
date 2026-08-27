import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(URL)

print(df.shape)
print(df.info())
print(df.isna().sum())
print(df.duplicated().sum())

df = df.drop_duplicates()

for col in ["survived", "pclass", "age", "sibsp", "parch", "fare"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["deck"] = df["deck"].fillna("Unknown")
df["fare"] = df["fare"].fillna(df["fare"].median())

df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"] = (df["family_size"] == 1).astype(int)

print(df.describe(include="all"))
print(df.groupby("sex")["survived"].mean())
print(df.groupby("pclass")["survived"].mean())

plt.hist(df["age"], bins=20)
plt.title("Passenger Age Distribution")
plt.xlabel("Age"); plt.ylabel("Count"); plt.show()
