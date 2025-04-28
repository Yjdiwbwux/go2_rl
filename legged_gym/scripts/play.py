import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
from scipy.interpolate import interp1d

points = torch.tensor([
    [0.06, -0.35],  # 起点
    [0.2, -0.3],    # 中间点
    [0.25, -0.15],  # 终点
    [0.2, -0.3],    # 再次经过中间点（返回路径）
    [0.06, -0.35]   # 回到起点
])
num_intermediate_points = 100  # 每段插值点数
total_steps = 1100             # 总步数
t_keyframes = torch.linspace(0, 1, len(points))
t_interp = torch.linspace(0, 1, num_intermediate_points * (len(points) - 1))

# 对 x 和 y 分别进行二次插值
interp_x = interp1d(t_keyframes.numpy(), points[:, 0].numpy(), kind='quadratic')
interp_y = interp1d(t_keyframes.numpy(), points[:, 1].numpy(), kind='quadratic')

# 生成插值后的完整路径（包含往返）
full_path = torch.stack([
    torch.tensor(interp_x(t_interp.numpy())),
    torch.tensor(interp_y(t_interp.numpy()))
], dim=1)

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 20)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    print(obs.shape)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    
    for i in range(int(10*int(env.max_episode_length))):
        # for env_ids in range(env.num_envs):
        #     idx = env.episode_length_buf[env_ids].to('cpu') % len(full_path)
        #     position = full_path[idx]
        #     obs.detach()[env_ids,13] = position[0].item()
        #     obs.detach()[env_ids,15] = position[1].item()
        #     print(position)
        actions = policy(obs.detach())      
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
