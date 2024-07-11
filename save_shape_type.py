# 丸,三角,逆三角,四角,八角で分類するファイル
import pandas as pd

shape_dict = {"circle":[0,1,2,3,4,5,6,7,8,9,10,15,16,17,32,33,34,35,36,37,38,39,40,41,42],"triangle":[11,18,19,20,21,22,23,24,25,26,27,28,29,30,31],"r_triangle":[13],"square":[12],"octagon":[14]}

def classify_shape(id):
    for shape in shape_dict.keys():
        if id in shape_dict[shape]:
            return shape


if __name__=="__main__":
    #meta_train_path = "../data/dataset/GTSRB/Train.csv"
    meta_test_path = "../data/dataset/GTSRB/Test.csv"
    df = pd.read_csv(meta_test_path)
    columns = df.columns
    for col in columns:
        if "Unnamed" in col:
            df = df.drop(col,axis=1)
    df["shape"] = df["ClassId"].apply(classify_shape)
    df.to_csv(meta_test_path,index=False)
    

