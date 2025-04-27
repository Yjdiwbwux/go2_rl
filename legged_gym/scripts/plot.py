import torch
import numpy as np
import matplotlib.pyplot as plt

# Initial setup
start_point = torch.tensor([0.06, -0.35])  # Start point (0.06, -0.35)
end_point = torch.tensor([0.3, -0.1])      # End point (0.3, -0.1)
num_intermediate_points = 100              # Number of interpolation points
total_steps = 1000                         # Total steps

# Generate linear interpolation points from start to end
t = torch.linspace(0, 1, num_intermediate_points)
forward_path = start_point + t.unsqueeze(1) * (end_point - start_point)

# Generate path back from end to start
backward_path = end_point + t.unsqueeze(1) * (start_point - end_point)

# Combine full path (forward + backward, avoiding duplicate endpoint)
full_path = torch.cat([forward_path, backward_path[1:]])

# Initialize position and storage for plotting
position = start_point.clone()
path_length = full_path.shape[0]
trajectory_x = []
trajectory_y = []

plt.figure(figsize=(8, 6))
plt.xlim(min(start_point[0], end_point[0])-0.1, max(start_point[0], end_point[0])+0.1)
plt.ylim(min(start_point[1], end_point[1])-0.1, max(start_point[1], end_point[1])+0.1)
plt.grid(True)
plt.title("Position Trajectory")

for step in range(total_steps):
    # Calculate current path index (cyclic)
    idx = step % path_length
    
    # Update position
    position = full_path[idx]
    
    # Store position for plotting
    trajectory_x.append(position[0].item())
    trajectory_y.append(position[1].item())
    
    # Update plot every 50 steps for performance
    if step % 50 == 0 or step == total_steps-1:
        plt.clf()
        plt.plot(trajectory_x, trajectory_y, 'b-', alpha=0.5, label='Trajectory')
        plt.plot(position[0].item(), position[1].item(), 'ro', label='Current Position')
        plt.plot(start_point[0], start_point[1], 'go', label='Start')
        plt.plot(end_point[0], end_point[1], 'yo', label='End')
        plt.xlim(min(start_point[0], end_point[0])-0.1, max(start_point[0], end_point[0])+0.1)
        plt.ylim(min(start_point[1], end_point[1])-0.1, max(start_point[1], end_point[1])+0.1)
        plt.grid(True)
        plt.title(f"Position Trajectory (Step {step}/{total_steps})")
        plt.legend()
        plt.pause(0.1)

# Final static plot
plt.show()
print(f"Final Position: {position.numpy()}")