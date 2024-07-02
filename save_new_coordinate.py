import pandas as pd

def scale(row):
    w = row["Width"]
    h = row["Height"]
    x1 = row["Roi.X1"]
    y1 = row["Roi.Y1"]
    x2 = row["Roi.X2"]
    y2 = row["Roi.Y2"]
    w_scale = 32/w
    h_scale = 32/h
    new_x1 = (w_scale*x1).round().astype(int)
    new_y1 = (h_scale*y1).round().astype(int)
    new_x2 = (w_scale*x2).round().astype(int)
    new_y2 = (h_scale*y2).round().astype(int)
    return new_x1,new_y1,new_x2,new_y2

meta_train_path = "../data/dataset/GTSRB/Test.csv"
df = pd.read_csv(meta_train_path)

columns = df.columns
for col in columns:
    if "Unnamed" in col:
        df = df.drop(col,axis=1)

df["new_x1"], df["new_y1"], df["new_x2"], df["new_y2"] = 0, 0, 0, 0
for index in df.index:
    df["new_x1"][index], df["new_y1"][index], df["new_x2"][index], df["new_y2"][index] = scale(df.iloc[index])

df.to_csv(meta_train_path,index=False)