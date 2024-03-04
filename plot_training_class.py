# トレーニングデータ可視化のためのpythonファイル

import pickle
import matplotlib.pyplot as plt

with open('../data/dataset/GTSRB/train.pkl', 'rb') as f:
    train = pickle.load(f)
    train_x, train_y = train['data'], train['labels']
with open('../data/dataset/GTSRB/test.pkl', 'rb') as f:
    test = pickle.load(f)
    test_x, test_y = test['data'], test['labels']
count = dict()
for label in train_y:
    if label in count.keys():
        count[label] += 1
    else:
        count[label] = 1
print((count))

fig, ax = plt.subplots()
ax.bar(range(len(count)),count.values())
fig.savefig('../data/training_class.png')
plt.close(fig)