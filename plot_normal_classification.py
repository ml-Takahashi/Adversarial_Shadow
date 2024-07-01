import matplotlib.pyplot as plt
import pickle


count_dict_path = "../data/dict/count_test_images.pkl"
normal_acc_path = "../data/dict/count_normal_acc.pkl"

with open(count_dict_path, 'rb') as f:
        test_count = pickle.load(f)

with open(normal_acc_path, 'rb') as f:
        normal_acc_dict = pickle.load(f)

# 値を取得
count_values = list(test_count.values())
acc_values = list(normal_acc_dict.values())


# 棒グラフの幅
bar_width = 0.4

# 棒グラフをプロット
plt.bar(list(test_count.keys()), count_values, color='blue', width=bar_width, label='test_data')
plt.bar(normal_acc_dict.keys(), acc_values, color='red', width=bar_width, label='correct classification')

# グラフのラベルとタイトルを設定
plt.xlabel('Keys', fontweight='bold')
plt.ylabel('Values', fontweight='bold')
plt.title('Comparison of Two Dictionaries')

# x軸の目盛りを設定
plt.xticks(range(43), range(43), fontsize=6)
# 凡例を表示
plt.legend()

# グラフを画像として保存
plt.savefig('comparison_of_two_dictionaries.png')

# グラフを表示
plt.show()
