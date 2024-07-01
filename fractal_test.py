import matplotlib.pyplot as plt
import numpy as np

def draw_ginkgo_leaf(center, radius, depth):
    if depth == 0:
        # 基本的な扇形を描画
        angles = np.linspace(0, np.pi, 100)
        x_vals = [center[0] + radius * np.cos(angle) for angle in angles]
        y_vals = [center[1] + radius * np.sin(angle) for angle in angles]

        # 扇形の先端を分割して模様を追加
        split_point_x = center[0] + radius * np.cos(np.pi / 2)
        split_point_y = center[1] + radius * np.sin(np.pi / 2)
        split_left_x = split_point_x - radius * 0.1
        split_right_x = split_point_x + radius * 0.1

        # 分割した点をリストに追加して扇形を描画
        x_vals += [split_left_x, split_point_x, split_right_x]
        y_vals += [split_point_y, split_point_y + radius * 0.15, split_point_y]

        plt.fill(x_vals + [center[0]], y_vals + [center[1]], 'green')
    else:
        # 再帰的に小さな葉を追加
        draw_ginkgo_leaf(center, radius * 0.9, depth - 1)  # 扇形を縮小して描画

def main():
    plt.figure(figsize=(6, 6))
    draw_ginkgo_leaf((0, 0), 100, 3)  # 中心位置, 初期半径, 再帰の深さ
    plt.axis('equal')
    plt.axis('off')
    plt.show()

main()