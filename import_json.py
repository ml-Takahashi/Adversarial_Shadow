import json

# json.load関数を使ったjsonファイルの読み込み
json_path = "../data/dataset/GTSRB/annotation_Train/0/00000_00000_00000.json"
with open(json_path) as f:
    di = json.load(f)

print(di["shapes"])  # deep insider：キーを指定して値を取得
