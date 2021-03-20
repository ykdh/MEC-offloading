"""
移动设备，任务，基站
"""

# 移动设备：计算能力MHz，
import random
import numpy as np


def get_md_info(num):
    md = {}
    i = 0
    while i < num:
        md[i] = MD(random.randint(800, 900), 23)
        i += 1
    return md


class MD:
    def __init__(self, cpu_frequency, transmitting_power):
        self.cpu_frequency = cpu_frequency
        self.transmitting_power = transmitting_power


# 任务：设备编号，数据量Mbit，cpu工作量MCycle，时延约束
def get_task_info(num):
    task = {}
    i = 0
    while i < num:
        task[i] = Task(i, random.uniform(0.3, 0.5), random.randint(900, 1100), 1.2)
        i += 1
    return task


class Task:
    def __init__(self, md, data_size, cpu_cycles, delay_constraints):
        self.md = md
        self.data_size = data_size
        self.cpu_cycles = cpu_cycles
        self.delay_constraints = delay_constraints


# 基站：计算能力MHz
def get_bs_info(num):
    bs = {}
    i = 0
    while i < num:
        bs[i] = BS(random.randint(2800, 3200))
        i += 1
    return bs


class BS:
    def __init__(self, cpu_frequency):
        self.cpu_frequency = cpu_frequency
