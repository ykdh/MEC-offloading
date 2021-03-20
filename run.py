from environment import ENV
from network import Agent
import numpy as np
import matplotlib.pyplot as plt

num_md = 15
num_task = 15
num_bs = 3

env = ENV(num_md, num_task, num_bs)

# 参数？？
# UEnet = Agent(alpha=0.0005, beta=0.005, input_dims=8, tau=0.01, \
#               env=None, batch_size=64, layer1_size=500, layer2_size=300,
#               n_actions=1)
MECSnet = Agent(alpha=0.0004, beta=0.004, input_dims=num_task * 3 + num_bs,
                tau=0.01, env=env, batch_size=64, layer1_size=500,
                layer2_size=300, n_actions=3)

# alpha beta tau batch_size

score_record = []
score_record_step = []
count_record = []
count_record_step = []
time_record = []
time_record_step = []
for i in range(800):
    done = False
    score = 0
    obs = env.get_init_state()
    # 没分配完
    while not done:
        act = MECSnet.choose_action(obs)
        new_state, reward, done = env.step(act)
        MECSnet.remember(obs, act, reward, new_state, int(done))
        MECSnet.learn()
        score += reward
        obs = new_state
        # print('reward is： {}'.format(reward))

    # 本轮的reward追加到list中
    score_record.append(score)
    # print('episode ', i, 'score %.2f' % score, '100 game average %.2f' % np.mean(score[-100:]))
    print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
    count_record.append(1 - env.count_wrong / num_task)
    time_record.append(env.time)
    if i % 25 == 0:
        # UEnet.save_models()
        MECSnet.save_models()
        score_record_step.append(np.mean(score_record))
        count_record_step.append(np.mean(count_record))
        time_record_step.append(np.mean(time_record))

# reward
plt.figure()
x_data = range(len(score_record))
plt.plot(x_data, score_record)

plt.figure()
x_data = range(len(score_record_step))
plt.plot(x_data, score_record_step)

# 卸载成功率
plt.figure()
x_data = range(len(count_record))
plt.plot(x_data, count_record)

plt.figure()
x_data = range(len(count_record_step))
plt.plot(x_data, count_record_step)

# 每回合时延
plt.figure()
x_data = range(len(time_record))
plt.plot(x_data, time_record)

plt.figure()
x_data = range(len(time_record_step))
plt.plot(x_data, time_record_step)
plt.show()

# print("ddpg: ", np.mean(score_record[-100]))
# print("local: ", env.all_local())
