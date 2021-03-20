# 环境
"""
状态：卸载决策，当前进度，分配计算资源，剩余计算资源
动作：是否卸载，卸载到哪，需要计算资源
奖励：不满足约束为-T，满足约束为T-t
"""
from other import *
import numpy as np
from _collections import deque


# 环境：移动设备数量，任务数量，基站数量
class ENV:
    def __init__(self, num_md, num_task, num_bs):
        self.num_md = num_md
        self.num_task = num_task
        self.num_bs = num_bs
        self.md = get_md_info(num_md)
        self.task = get_task_info(num_task)
        self.bs = get_bs_info(num_bs)
        # print(len(self.md), len(self.task), len(self.bs))
        self.fmax = 3200  # 临时的！！MEC服务器最大计算资源 MHz
        self.is_off = [0] * num_task  # 卸载决策
        self.alloc_resource = [0] * num_task  # 分配的资源
        self.rest_resource = [0] * num_bs
        # 剩余的资源
        i = 0
        while i < num_bs:
            self.rest_resource[i] = self.bs[i].cpu_frequency
            i += 1
        self.progress = [0] * num_task  # 进度
        self.reward = 0
        self.done = False
        self.waiting = deque()
        self.new_queue = 0
        self.count_wrong = 0
        self.time = 0

    def get_init_state(self):
        self.count_wrong = 0
        self.time = 0
        self.is_off = [0] * self.num_task  # 卸载决策
        self.alloc_resource = [0] * self.num_task  # 分配的资源
        # 剩余的资源
        i = 0
        while i < self.num_bs:
            self.rest_resource[i] = self.bs[i].cpu_frequency
            i += 1
        self.progress = [0] * self.num_task  # 进度
        self.new_task()
        # np.array(self.is_off).astype(int)
        # np.array(self.alloc_resource).astype(int)
        # np.array(self.rest_resource).astype(int)
        # np.array(self.progress).astype(int)
        # state = np.concatenate((self.is_off, self.alloc_resource, self.rest_resource, self.progress)).reshape((4, 10))
        state = np.concatenate((self.is_off, self.progress, self.alloc_resource, self.rest_resource))
        return state

    def new_task(self):
        self.waiting.clear()
        # 待处理任务个数
        rest = self.num_task - sum(self.progress)
        length = random.randint(1, self.num_bs) if rest >= self.num_bs else rest
        # 该从几号任务开始了
        task_id = sum(self.progress)
        self.waiting.extend(range(task_id, task_id + length))
        self.new_queue = 1

    def all_local(self):
        i = 1
        cost = 0
        while i <= self.num_task:
            cost += self.task[i].cpu_cycles / self.md[self.task[i].md].cpu_frequency
            i += 1
        return cost

    def step(self, action):
        # 是否满足资源和时延约束 [0]是否卸载，[1]卸载到哪，[2]需要的计算资源
        # 噪声的影响
        i = 0
        while i < 3:
            if action[i] > 1: action[i] = 1
            if action[i] < -1: action[i] = -1
            i += 1

        # 从action获得数据
        get1 = 1 if action[0] > 0 else 0
        get2 = int((action[1] + 1) * (self.num_bs - 1) / 2)
        get3 = int((action[2] + 1) * 2000 / 2) + 1000
        # print("get123: ", get1, get2, get3)

        i_task = self.waiting[0]
        i_md = self.task[i_task].md
        T = self.task[i_task].delay_constraints

        if self.new_queue == 1:
            i = 0
            while i < self.num_bs:
                self.rest_resource[i] = self.bs[i].cpu_frequency
                i += 1
        # 本地
        if get1 == 0:
            f = self.md[i_md].cpu_frequency
            t = self.task[i_task].cpu_cycles / f
            # 满足约束
            if t <= T:
                self.is_off[i_task] = 0
                self.progress[i_task] = 1
                self.alloc_resource[i_task] = f
                self.done = True if sum(self.progress) == self.num_task else False
                reward = T - t
                self.waiting.popleft()
                self.time += t
            else:
                # A方案
                # reward = -1
                # self.done = False
                # B方案
                # reward = 0
                # i = 0
                # while i < self.num_task:
                #     if self.progress[i] == 0:
                #         reward += -1
                #         self.count_wrong += 1
                #     i += 1
                # self.done = True
                # C方案
                self.waiting.popleft()
                reward = -T
                self.progress[i_task] = 1
                self.done = True if sum(self.progress) == self.num_task else False
                self.count_wrong += 1
                self.time += T

        # 卸载
        else:
            f = get3
            v = 2  # Mb/s
            t = self.task[i_task].data_size / v + self.task[i_task].cpu_cycles / f
            # 满足约束
            if f <= self.rest_resource[get2] and t <= T:
                self.is_off[i_task] = get2 + 1
                self.progress[i_task] = 1
                self.alloc_resource[i_task] = f
                self.rest_resource[get2] -= f
                self.done = True if sum(self.progress) == self.num_task else False
                reward = T - t
                self.waiting.popleft()
                self.time += t
            else:
                # A方案
                # reward = -1
                # self.done = False
                # B方案
                # reward = 0
                # i = 0
                # while i < self.num_task:
                #     if self.progress[i] == 0:
                #         reward += -1
                #         self.count_wrong += 1
                #     i += 1
                # self.done = True
                # C方案
                self.waiting.popleft()
                reward = -T
                self.progress[i_task] = 1
                self.done = True if sum(self.progress) == self.num_task else False
                self.count_wrong += 1
                self.time += T
        if self.waiting:
            self.new_queue = 0
        else:
            self.new_task()
            self.new_queue = 1
        state = np.concatenate((self.is_off, self.progress, self.alloc_resource, self.rest_resource))
        return state, reward, self.done
