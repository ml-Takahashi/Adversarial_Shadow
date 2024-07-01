import pandas as pd

df = pd.read_csv("../data/dataset/GTSRB/Test.csv")
df["ClassId"] = df["ClassId"].astype("int")
df = df.sort_values("ClassId")
df.to_csv("../data/dataset/GTSRB/Test.csv")

df = pd.read_csv("../data/dataset/GTSRB/Train.csv")
df["ClassId"] = df["ClassId"].astype("int")
df = df.sort_values("ClassId")
df.to_csv("../data/dataset/GTSRB/Train.csv")