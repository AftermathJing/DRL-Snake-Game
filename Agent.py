#!/usr/bin/python
# -*- coding: utf-8 -*-
import pygame
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from copy import deepcopy
from core import *


class AgentPPO:
    """
    使用PPO（Proximal Policy Optimization）算法
    主要用于强化学习中的策略优化
    """
    def __init__(self):
        """
        初始化PPO智能体的参数。
        """
        super().__init__()
        self.ratio_clip = 0.2  # PPO的裁剪比率
        self.lambda_entropy = 0.02  # 熵正则化项的系数
        self.lambda_gae_adv = 0.98  # 广义优势估计(GAE)的衰减系数
        self.get_reward_sum = None  # 计算回报总和的函数，取决于是否使用GAE

        self.state = None  # 当前状态
        self.device = None  # 运行设备（CPU或GPU）
        self.criterion = None  # 损失函数
        self.act = self.act_optimizer = None  # 行为模型及其优化器
        self.cri = self.cri_optimizer = self.cri_target = None  # 评价模型、其优化器及目标模型

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False):
        """
        初始化智能体的网络和优化器。
        :param net_dim: 网络维度
        :param state_dim: 状态维度
        :param action_dim: 动作维度
        :param learning_rate: 学习率
        :param if_use_gae: 是否使用广义优势估算
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

        # 初始化行为和评价网络
        self.act = ActorPPO(net_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(net_dim, state_dim).to(self.device)
        self.cri_target = deepcopy(self.cri) if self.cri_target is True else self.cri

        self.criterion = torch.nn.SmoothL1Loss()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)

    def select_action(self, state):
        """
        根据当前状态选择动作。
        :param state: 当前状态
        :return: 选择的动作和噪声
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(states)
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()

    def explore_env(self, env, target_step, reward_scale, gamma):
        """
        在环境中执行动作并收集数据。
        :param env: 环境对象
        :param target_step: 探索的步数
        :param reward_scale: 奖励缩放因子
        :param gamma: 折扣因子
        :return: 轨迹列表，包含每一步的状态和奖励信息
        """
        trajectory_list = list()

        state = self.state
        for _ in range(target_step):
            action, noise = self.select_action(state)
            next_state, reward, done, _ = env.step(np.tanh(action))
            other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
            trajectory_list.append((state, other))

            state = env.reset() if done else next_state
        self.state = state
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        更新网络参数，包括行为网络和评价网络。
        :param buffer: 数据缓冲区
        :param batch_size: 批量大小
        :param repeat_times: 更新次数
        :param soft_update_tau: 软更新参数
        """
        buffer.update_now_len()
        buf_len = buffer.now_len
        buf_state, buf_action, buf_r_sum, buf_logprob, buf_advantage = self.prepare_buffer(buffer)
        buffer.empty_buffer()

        '''PPO: Trust Region的替代目标'''
        obj_critic = obj_actor = logprob = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optimizer, obj_actor)

            value = self.cri(state).squeeze(1)
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        return obj_critic.item(), obj_actor.item(), logprob.mean().item()

    def prepare_buffer(self, buffer):
        """
        从缓冲区准备数据，用于网络训练。
        :param buffer: 缓冲区
        """
        buf_len = buffer.now_len

        with torch.no_grad():  # 计算反向奖励
            reward, mask, action, a_noise, state = buffer.sample_all()

            bs = 2 ** 10  # 如果GPU内存不足，设置较小的批量大小
            value = torch.cat([self.cri_target(state[i:i + bs]) for i in range(0, state.size(0), bs)], dim=0)
            logprob = self.act.get_old_logprob(action, a_noise)

            pre_state = torch.as_tensor((self.state,), dtype=torch.float32, device=self.device)
            pre_r_sum = self.cri(pre_state).detach()
            r_sum, advantage = self.get_reward_sum(self, buf_len, reward, mask, value, pre_r_sum)
        return state, action, r_sum, logprob, advantage

    @staticmethod
    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value, pre_r_sum) -> (torch.Tensor, torch.Tensor):
        """
        计算原始奖励总和。
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)

        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    @staticmethod
    def get_reward_sum_gae(self, buf_len, buf_reward, buf_mask, buf_value, pre_r_sum) -> (torch.Tensor, torch.Tensor):
        """
        使用广义优势估计(GAE)计算奖励总和。
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)

        pre_advantage = 0  # 上一步的优势值
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + buf_mask[i] * (pre_advantage - buf_value[i])
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    @staticmethod
    def optim_update(optimizer, objective):
        """
        执行优化器的更新。
        :param optimizer: 优化器
        :param objective: 优化目标
        """
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """
        执行软更新，逐渐合并当前网络和目标网络的权重。
        :param target_net: 目标网络
        :param current_net: 当前网络
        :param tau: 更新系数
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))


class AgentDiscretePPO(AgentPPO):
    """
    AgentPPO类的子类，专门用于处理离散动作空间的PPO算法实现
    """
    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False):
        """
        初始化离散动作空间的智能体的网络和优化器。
        :param net_dim: 网络中隐藏层的维度
        :param state_dim: 状态的维度
        :param action_dim: 动作的维度
        :param learning_rate: 学习率
        :param if_use_gae: 是否使用广义优势估计（Generalized Advantage Estimation）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

        # 初始化动作生成器和评价器，适用于离散动作空间
        self.act = ActorDiscretePPO(net_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(net_dim, state_dim).to(self.device)
        self.cri_target = deepcopy(self.cri) if self.cri_target is True else self.cri

        self.criterion = torch.nn.SmoothL1Loss()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)

    def explore_env(self, env, target_step, reward_scale, gamma):
        """
        探索环境，收集训练数据。
        :param env: 与智能体交互的环境对象
        :param target_step: 在环境中执行的步数
        :param reward_scale: 奖励缩放因子
        :param gamma: 折扣因子，用于计算未来奖励的当前价值
        :return: 返回收集的轨迹列表，每个元素包含状态和动作的信息
        """
        trajectory_list = list()

        state = self.state
        for _ in range(target_step):
            # 加快训练速度，不渲染动画
            # env.render()
            # for event in pygame.event.get():  # 不加这句render要卡，不清楚原因
            #     pass

            # 选择动作，得到动作的整数表示及其概率
            a_int, a_prob = self.select_action(state)
            # 在环境中执行动作，获取新的状态和奖励
            next_state, reward, done, _ = env.step(int(a_int))
            other = (reward * reward_scale, 0.0 if done else gamma, a_int, *a_prob)
            trajectory_list.append((state, other))

            # 如果完成，则重置环境；否则继续在新状态下执行
            state = env.reset() if done else next_state
        self.state = state
        return trajectory_list
