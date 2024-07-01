import numpy as np
import matplotlib.pyplot as plt

def random_walk(steps):
    # x, y の位置を記録する配列
    x, y = [0], [0]
    
    # ランダムウォークを実行
    for _ in range(steps):
        dx, dy = np.random.choice([-1, 0, 1], 2)  # x, y それぞれに -1, 0, 1 の変化量をランダムに選ぶ
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
    
    return x, y

# ランダムウォークのステップ数
num_steps = 500000

# ランダムウォークを実行
x, y = random_walk(num_steps)

# 描画設定
plt.figure(figsize=(0.5, 0.5), dpi=100)  # 実際の画像サイズとして0.5x0.5インチに設定し、DPIを100に設定することで50x50ピクセルの画像を生成
plt.plot(x, y, color='black', linewidth=0.5)  # 黒色で経路を描画、線の太さを調整
plt.xlim(-25, 25)  # x軸の範囲を設定
plt.ylim(-25, 25)  # y軸の範囲を設定
plt.axis('equal')
plt.axis('off')  # 軸を表示しない

# 画像として保存
plt.savefig('random_walk_50x50.png', bbox_inches='tight', pad_inches=0)
plt.show()
