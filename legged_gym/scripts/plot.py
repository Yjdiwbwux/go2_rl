import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 定义关键点：起点 → 终点 → 起点（直线往返）
points = torch.tensor([
    [0.0, 0.2],   # 起点
    [0.0, 0.35],  # 终点
    [0.0, 0.2]    # 返回起点
])

num_intermediate_points = 200  # 每段插值点数
total_steps = 1000            # 总步数

# 生成插值参数 t（从0到1，覆盖所有关键点）
t_keyframes = torch.linspace(0, 1, len(points))
t_interp = torch.linspace(0, 1, num_intermediate_points * (len(points) - 1))

# 对 x 和 y 分别进行线性插值（kind='linear'）
interp_x = interp1d(t_keyframes.numpy(), points[:, 0].numpy(), kind='linear')
interp_y = interp1d(t_keyframes.numpy(), points[:, 1].numpy(), kind='linear')

# 生成插值后的完整路径（直线往返）
full_path = torch.stack([
    torch.tensor(interp_x(t_interp.numpy())),
    torch.tensor(interp_y(t_interp.numpy()))
], dim=1)

# 初始化位置和轨迹存储
position = points[0].clone()
trajectory_x, trajectory_y = [], []

# 绘图设置
plt.figure(figsize=(8, 6))
plt.grid(True)
plt.title("Linear Interpolation (Straight Line Round Trip)")

for step in range(total_steps):
    idx = step % len(full_path)
    position = full_path[idx]
    trajectory_x.append(position[0].item())
    trajectory_y.append(position[1].item())

    if step % 5 == 0 or step == total_steps - 1:
        plt.clf()
        plt.plot(full_path[:, 0], full_path[:, 1], 'k--', alpha=0.3, label='Interpolated Path')
        plt.plot(trajectory_x, trajectory_y, 'b-', alpha=0.5, label='Trajectory')
        plt.plot(position[0], position[1], 'ro', label='Current Position')
        plt.scatter(points[:, 0], points[:, 1], c=['g', 'y', 'g'], s=100, label='Key Points')
        plt.xlim(-0.05, 0.05)  # x范围固定（因为是直线）
        plt.ylim(0.15, 0.4)    # y范围调整
        plt.grid(True)
        plt.title(f"Step {step}/{total_steps}")
        plt.legend()
        plt.pause(0.01)

plt.show()
print(f"Final Position: {position.numpy()}")