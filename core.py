#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import numpy as np


class ActorPPO(nn.Module):
    """
    实现 PPO（Proximal Policy Optimization）算法中的行为网络。
    一个典型的PPO行为网络，其中包括动作的生成、对数概率和策略熵的计算等关键功能。
    通过这种方式，ActorPPO 可以在给定状态的基础上生成相应的动作，同时计算这些动作的概率属性。
    """
    def __init__(self, mid_dim, state_dim, action_dim):
        """
        初始化行为模型。
        :param mid_dim: 网络中间层的维度。
        :param state_dim: 输入状态的维度。
        :param action_dim: 输出动作的维度。
        """
        super().__init__()
        # 定义神经网络结构，包括线性层和激活函数
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),  # 第一层
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),  # 第二层
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(),  # 第三层
            nn.Linear(mid_dim, action_dim)  # 输出层
        )
        layer_norm(self.net[-1], std=0.1)  # 对输出层应用层归一化

        # 初始化动作的对数标准差为可学习的参数
        self.a_logstd = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        # 预计算正态分布的常数部分
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        """
        定义前向传播。
        :param state: 输入的状态
        :return: 通过tanh激活函数限制在[-1, 1]范围内的动作
        """
        return self.net(state).tanh()  # 输出动作值

    def get_action(self, state):
        """
        根据当前状态生成动作和对应的噪声。
        :param state: 当前状态
        :return: 动作和噪声
        """
        a_avg = self.net(state)  # 计算动作的平均值
        a_std = self.a_logstd.exp()  # 计算动作的标准差

        noise = torch.randn_like(a_avg)  # 生成噪声
        action = a_avg + noise * a_std  # 添加噪声，生成实际动作
        return action, noise

    def get_logprob_entropy(self, state, action):
        """
        计算给定状态和动作的对数概率和策略熵。
        :param state: 当前状态
        :param action: 执行的动作
        :return: 对数概率和策略熵
        """
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # 计算对数概率

        dist_entropy = (logprob.exp() * logprob).mean()  # 计算策略熵
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):
        """
        计算旧的对数概率，通常用于算法中的重要性采样部分。
        :param _action: 旧的动作值
        :param noise: 动作和平均值之间的噪声
        :return: 旧的对数概率
        """
        delta = noise.pow(2) * 0.5
        return -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # 计算旧的对数概率


class ActorDiscretePPO(nn.Module):
    """
    实现离散动作空间中的PPO算法的行为网络
    提供了在离散动作空间中使用PPO算法所需的核心功能，包括动作的生成、对数概率的计算以及策略熵的计算
    """
    def __init__(self, mid_dim, state_dim, action_dim):
        """
        初始化离散行为模型。
        :param mid_dim: 网络中间层的维度。
        :param state_dim: 输入状态的维度。
        :param action_dim: 输出动作的维度，即动作空间的大小。
        """
        super().__init__()
        # 构建神经网络，包含三个隐藏层和一个输出层
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
            nn.Linear(mid_dim, action_dim)
        )
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=-1)  # softmax函数，用于将网络输出转换为概率
        self.Categorical = torch.distributions.Categorical  # 用于创建分类分布的工具

    def forward(self, state):
        """
        定义前向传播。
        :param state: 输入的状态
        :return: 未经softmax处理的动作概率
        """
        return self.net(state)  # 返回网络的原始输出

    def get_action(self, state):
        """
        根据当前状态生成动作。
        :param state: 当前状态
        :return: 选择的动作和动作的概率
        """
        a_prob = self.soft_max(self.net(state))  # 计算动作概率
        samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)  # 从概率分布中采样动作
        action = samples_2d.reshape(state.size(0))  # 调整形状以匹配状态批次大小
        return action, a_prob

    def get_logprob_entropy(self, state, action):
        """
        计算给定状态和动作的对数概率和策略熵。
        :param state: 当前状态
        :param action: 执行的动作
        :return: 对数概率和策略熵
        """
        a_prob = self.soft_max(self.net(state))  # 计算动作概率
        dist = self.Categorical(a_prob)  # 创建对应的分类分布
        a_int = action.squeeze(1).long()  # 确保动作是长整型，并适配批次形状
        return dist.log_prob(a_int), dist.entropy().mean()  # 计算对数概率和平均熵

    def get_old_logprob(self, action, a_prob):
        """
        计算旧动作在新的动作概率下的对数概率，常用于算法中的重要性采样。
        :param action: 旧动作
        :param a_prob: 当前的动作概率
        :return: 旧动作的对数概率
        """
        dist = self.Categorical(a_prob)  # 使用当前动作概率创建分类分布
        return dist.log_prob(action.long().squeeze(1))  # 计算旧动作的对数概率


class CriticAdv(nn.Module):
    """
    实现评价网络（Critic Network），该网络在PPO算法中用于评估状态的价值（Q值）
    """
    def __init__(self, mid_dim, state_dim):
        """
        初始化评价网络。
        :param mid_dim: 网络中间层的维度。
        :param state_dim: 输入状态的维度。
        """
        super().__init__()
        # 构建神经网络，包含三个隐藏层和一个输出层
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),  # 第一层
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),  # 第二层
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(),  # 第三层
            nn.Linear(mid_dim, 1)  # 输出层，输出单一的Q值
        )
        layer_norm(self.net[-1], std=0.5)  # 对输出层应用层归一化，设置标准差为0.5

    def forward(self, state):
        """
        定义前向传播。
        :param state: 输入的状态
        :return: 状态的Q值，表示状态的价值
        """
        return self.net(state)  # 返回计算得到的Q值


class ReplayBuffer:
    """
    用于强化学习中的经验回放机制
    存储智能体与环境交互产生的数据（如状态、动作、奖励等），并提供抽样这些数据的方法以用于训练
    这个缓冲区类通过提供添加、扩展、抽样和清空数据的方法，支持在强化学习训练过程中有效地管理和使用经验数据
    """
    def __init__(self, max_len, state_dim, action_dim, if_discrete):
        """
        初始化经验回放缓冲区。
        :param max_len: 缓冲区能存储的最大样本数量。
        :param state_dim: 状态的维度。
        :param action_dim: 动作的维度。
        :param if_discrete: 表示动作是否为离散类型。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len  # 缓冲区的最大长度
        self.now_len = 0  # 当前存储的数据长度
        self.next_idx = 0  # 下一个要写入的索引位置
        self.if_full = False  # 缓冲区是否已满
        self.action_dim = 1 if if_discrete else action_dim  # 如果动作是离散的，设置动作维度为1
        self.tuple = None
        self.np_torch = torch

        other_dim = 1 + 1 + self.action_dim + action_dim  # 定义其他数据的维度（奖励，结束标志，动作，动作噪声或概率）
        self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)  # 创建其他数据的缓冲区
        self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)  # 创建状态数据的缓冲区

    def append_buffer(self, state, other):
        """
        向缓冲区添加单个样本。
        :param state: 状态数据。
        :param other: 其他数据（包括奖励等）。
        """
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, other):
        """
        向缓冲区批量添加样本。
        :param state: 状态数据数组。
        :param other: 其他数据数组。
        """
        size = len(other)
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            # 如果超出缓冲区长度，则进行循环覆盖
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def extend_buffer_from_list(self, trajectory_list):
        """
        从轨迹列表中批量添加样本。
        :param trajectory_list: 轨迹列表，每个元素是一个(state, other)元组。
        """
        state_ary = np.array([item[0] for item in trajectory_list], dtype=np.float32)
        other_ary = np.array([item[1] for item in trajectory_list], dtype=np.float32)
        self.extend_buffer(state_ary, other_ary)

    def sample_batch(self, batch_size):
        """
        从缓冲区随机采样一批数据。
        :param batch_size: 批量大小。
        :return: 采样得到的批量数据。
        """
        indices = np.random.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (
            r_m_a[:, 0:1],  # 奖励
            r_m_a[:, 1:2],  # 结束标志，0.0表示结束，gamma表示未结束
            r_m_a[:, 2:],  # 动作数据
            self.buf_state[indices],  # 当前状态
            self.buf_state[indices + 1]  # 下一个状态
        )

    def sample_all(self):
        """
        从缓冲区采样所有数据。
        :return: 所有数据。
        """
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (
            all_other[:, 0],  # 奖励
            all_other[:, 1],  # 结束标志
            all_other[:, 2:2 + self.action_dim],  # 动作数据
            all_other[:, 2 + self.action_dim:],  # 动作噪声或概率
            torch.as_tensor(self.buf_state[:self.now_len], device=self.device)  # 状态
        )

    def update_now_len(self):
        """
        更新缓冲区中的当前数据长度。
        """
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer(self):
        """
        清空缓冲区。
        """
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
