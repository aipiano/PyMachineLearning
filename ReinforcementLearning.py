import numpy as np
from common import *
from abc import ABC
from abc import abstractmethod
from matplotlib import pyplot as plt
"""
强化学习：Sarsa, Q-Learning, 资格迹
目前适用于周期性任务
"""


class Environment(ABC):
    @abstractmethod
    def do(self, s, a):
        """
        在指定状态下执行动作a，返回下一个状态和奖励
        :param s: 当前状态（必须为可哈希的值，因为要作为字典的key）
        :param a: 要执行的动作
        :return: (s, r)
        """
        pass

    @abstractmethod
    def actions(self, s):
        """
        获取状态s下允许的动作
        :param s: 当前状态
        :return: list
        """
        pass

    @abstractmethod
    def reset(self, episodes, steps, rewords):
        """
        告知环境一个周期结束，并返回一个新的初始状态，进行新一轮的学习
        :param episodes: 已完成的周期数
        :param steps: 已执行的总步数
        :param rewords: 以获得的总奖励
        :return: state
        """
        pass


class ReinforcementLearning(ABC):
    def __init__(self, environment: Environment, q=None, init_value=0.0, e_greedy=0.02, step_size=0.1, discount=0.9, trace=0):
        """
        :param environment: 学习环境
        :param q: q值表（一个字典，key是状态，value是包含动作和值的字典）
        :param init_value: 遇到新状态时，q的初始值
        :param e_greedy: epsilon贪心选择法中的epsilon
        :param step_size: 学习步长(0,1)
        :param discount: 折扣(0,1]
        :param trace: 资格迹[0，1]，0代表不使用资格迹
        """
        self.environment = environment
        if q and isinstance(q, dict):
            self.q = q
            self.z = q.copy()
            for state in self.z:
                self.z[state] = 0
        else:
            self.q = {}
            self.z = {}   # eligibility traces
        self.init_value = init_value
        self.epsilon = e_greedy
        self.step_size = step_size
        self.discount = discount
        self.trace = trace

        self.episodes = 0
        self.rewords = 0
        self.steps = 0
        self.avg_rewords = 0.0
        self.avg_steps = 0.0

    def append_state(self, s):
        """
        若状态s不在q值表中，则加入其中并初始化
        :param s:
        :return: None
        """
        if s in self.q:
            return
        actions = self.environment.actions(s)
        self.q[s] = {a: self.init_value for a in actions} if actions else None
        if self.trace > 0:  # 若使用资格迹，则初始化之
            self.z[s] = {a: 0.0 for a in actions} if actions else None

    def max_action_value_on(self, s):
        """
        返回状态s下，最大的动作值
        :param s:
        :return: float
        """
        action_value_dict = self.q[s]
        if action_value_dict is None:
            return 0
        max_value = -np.inf
        for a, v in action_value_dict.items():
            max_value = max(max_value, v)
        return max_value

    def opt_action_on(self, s):
        """
        返回状态s下的最佳动作（即值最大的动作，若有多个动作对应最大值，则随机返回一个）
        :param s:
        :return: action
        """
        action_value_dict = self.q[s]
        if action_value_dict is None:
            return None
        max_value = self.max_action_value_on(s)
        opt_actions = [a for a, v in action_value_dict.items() if v == max_value]
        return random.choice(opt_actions)

    def e_greedy_sample(self, s):
        """
        在状态s下，使用e-greedy方法选取一个动作
        :param s:
        :return: action
        """
        if random.uniform(0, 1) < self.epsilon:
            actions = self.environment.actions(s)
            if actions is None:
                return None
            return random.choice(actions)
        else:
            return self.opt_action_on(s)

    def clear_traces(self):
        """
        将资格迹清零
        :return: None
        """
        if self.trace == 0:
            return
        for s in self.z:
            if self.z[s] is None:  # 若为terminal state，则跳过
                continue
            for a in self.z[s]:
                self.z[s][a] = 0.0

    def get_policy(self):
        """
        返回当前值表对应的最优策略
        :return: dict
        """
        policy = {}
        for s in self.q:
            policy[s] = self.opt_action_on(s)
        return policy

    def run_policy(self, max_episodes):
        """
        执行当前学到的最优策略，期间不会进行任何学习，若碰到未知的状态，则随机选择动作
        :param max_episodes:
        :return:
        """
        episodes = 0
        steps = 0
        rewords = 0

        # 初始化Q(s, a)
        init_state = self.environment.reset(episodes, steps, rewords)
        self.append_state(init_state)
        # 初始状态不能是terminal
        assert self.q[init_state] is not None

        # 选择最优动作
        current_state = init_state
        current_action = self.opt_action_on(init_state)
        while episodes < max_episodes:
            # 执行动作，获取下一个状态和奖励
            next_state, reword = self.environment.do(current_state, current_action)
            self.append_state(next_state)  # 如果是一个新的状态，会加入值表中
            steps += 1
            rewords += reword

            # 始终选择最优动作
            next_action = self.opt_action_on(next_state)
            if not next_action:  # 如果到达停止状态，则重置状态和动作，进行新一轮学习
                episodes += 1
                current_state = self.environment.reset(episodes, steps, rewords)
                current_action = self.opt_action_on(current_state)
                self.clear_traces()  # 清零资格迹
            else:
                current_state = next_state
                current_action = next_action

    @abstractmethod
    def run(self, max_episodes=500):
        """
        开始学习
        :param max_episodes:
        :return:
        """
        pass


class Sarsa(ReinforcementLearning):
    def run(self, max_episodes=500):
        # 初始化Q(s, a)
        init_state = self.environment.reset(self.episodes, self.steps, self.rewords)
        self.append_state(init_state)
        # 初始状态不能是terminal
        assert self.q[init_state] is not None

        # 使用e-greedy选择第一步动作
        current_state = init_state
        current_action = self.e_greedy_sample(init_state)
        while self.episodes < max_episodes:
            # 执行动作，获取下一个状态和奖励
            next_state, reword = self.environment.do(current_state, current_action)
            self.append_state(next_state)  # 如果是一个新的状态，会加入值表中
            self.steps += 1
            self.rewords += reword

            # 任然使用e-greedy选择下一步的动作
            next_action = self.e_greedy_sample(next_state)
            # 若没有下一步动作，说明当前状态是terminal state，则Q值为0
            qsa = self.q[next_state][next_action] if next_action else 0
            old_value = self.q[current_state][current_action]
            delta = reword + self.discount * qsa - old_value

            if self.trace == 0:  # 不使用资格迹，直接更新
                self.q[current_state][current_action] += self.step_size * delta
            else:   # 使用资格迹，这里实现的是replacing traces
                # 先将当前状态下的所有迹清零，然后对执行状态的迹置一（效果不明显）
                # for a in self.z[current_state]:
                #     self.z[current_state][a] = 0
                self.z[current_state][current_action] = 1
                for s in self.q:
                    if self.q[s] is None:  # 若为terminal state，则跳过
                        continue
                    for a in self.q[s]:
                        zsa = self.z[s][a]
                        if zsa < 1e-5:   # 资格迹过小就不更新，减少一点计算量
                            continue
                        self.q[s][a] += self.step_size * delta * zsa
                        self.z[s][a] *= self.discount * self.trace

            if not next_action:  # 如果到达停止状态，则重置状态和动作，进行新一轮学习
                self.episodes += 1
                self.avg_steps = self.steps / self.episodes
                self.avg_rewords = self.rewords / self.episodes
                self.clear_traces()  # 清零资格迹

                current_state = self.environment.reset(self.episodes, self.steps, self.rewords)
                self.append_state(current_state)  # 有可能返回一个之前没见过的初始状态
                current_action = self.e_greedy_sample(current_state)
            else:
                current_state = next_state
                current_action = next_action


class QLearning(ReinforcementLearning):
    def run(self, max_episodes=500):
        # 初始化Q(s, a)
        init_state = self.environment.reset(self.episodes, self.steps, self.rewords)
        self.append_state(init_state)
        # 初始状态不能是terminal
        assert self.q[init_state] is not None

        # 使用e-greedy选择第一步动作
        current_state = init_state
        current_action = self.e_greedy_sample(init_state)
        while self.episodes < max_episodes:
            # 执行动作，获取下一个状态和奖励
            next_state, reword = self.environment.do(current_state, current_action)
            self.append_state(next_state)  # 如果是一个新的状态，会加入值表中
            self.steps += 1
            self.rewords += reword

            # 任然使用e-greedy选择下一步的动作
            next_action = self.e_greedy_sample(next_state)
            # 但是对最优动作进行更新
            optimal_action = self.opt_action_on(next_state)
            # 若没有下一步动作，说明当前状态是terminal state，则Q值为0
            qsa = self.q[next_state][optimal_action] if next_action else 0
            old_value = self.q[current_state][current_action]
            delta = reword + self.discount * qsa - old_value

            if self.trace == 0:  # 不使用资格迹，直接更新
                self.q[current_state][current_action] += self.step_size * delta
            else:   # 使用资格迹，这里实现的是replacing traces
                # 先将当前状态下的所有迹清零，然后对执行状态的迹置一（效果不明显）
                # for a in self.z[current_state]:
                #     self.z[current_state][a] = 0
                self.z[current_state][current_action] = 1
                for s in self.q:
                    if self.q[s] is None:  # 若为terminal state，则跳过
                        continue
                    for a in self.q[s]:
                        zsa = self.z[s][a]
                        if zsa < 1e-5:  # 资格迹过小就不更新，减少一点计算量
                            continue
                        self.q[s][a] += self.step_size * delta * zsa
                        # 当一个探索（非最优）动作发生时，将所有资格迹置零
                        self.z[s][a] *= self.discount * self.trace if next_action == optimal_action else 0

            if not next_action:  # 如果到达停止状态，则重置状态和动作，进行新一轮学习
                self.episodes += 1
                self.avg_steps = self.steps / self.episodes
                self.avg_rewords = self.rewords / self.episodes
                self.clear_traces()  # 清零资格迹

                current_state = self.environment.reset(self.episodes, self.steps, self.rewords)
                self.append_state(current_state)  # 有可能返回一个之前没见过的初始状态
                current_action = self.e_greedy_sample(current_state)
            else:
                current_state = next_state
                current_action = next_action


class WindyGridWorld(Environment):
    def __init__(self, width, height, winds, reword_position):
        self.width = width
        self.height = height
        self.winds = winds
        self.start_position = (0, 0)
        self.reword_position = reword_position
        plt.ion()
        self.refresh_plot()

    def refresh_plot(self):
        plt.clf()
        plt.scatter(self.start_position[0], self.start_position[1], c='g', s=100)
        plt.scatter(self.reword_position[0], self.reword_position[1], c='b', s=100)

        plt.scatter(0, 0, s=1)
        plt.scatter(0, self.height - 1, s=1)
        plt.scatter(self.width - 1, 0, s=1)
        plt.scatter(self.width - 1, self.height - 1, s=1)

        plt.xticks([x for x in range(self.width)])
        plt.yticks([y for y in range(self.height)])
        plt.grid(which='both', linewidth=1, linestyle=':')

    def do(self, s, a):
        pos = list(s)

        if a == 'u':
            pos[1] += 1
        elif a == 'd':
            pos[1] -= 1
        elif a == 'l':
            pos[0] -= 1
        elif a == 'r':
            pos[0] += 1

        wind_force = self.winds[pos[0]] if pos[0] in self.winds else 0
        pos[1] += wind_force

        pos[0] = min(self.width - 1, max(0, pos[0]))
        pos[1] = min(self.height - 1, max(0, pos[1]))

        plt.plot([s[0], pos[0]], [s[1], pos[1]], '-r')
        plt.pause(0.001)

        reword = -1.0
        tpos = tuple(pos)
        if tpos == self.reword_position:
            reword = 1.0
        # elif tpos == self.start_position:
        #     reword = -1.0

        return tpos, reword

    def actions(self, s):
        if s == self.reword_position:
            return None
        x = s[0]
        y = s[1]
        actions = ['u', 'd', 'l', 'r']
        if x == 0:
            actions.remove('l')
        if x == self.width - 1:
            actions.remove('r')
        if y == 0:
            actions.remove('d')
        if y == self.height - 1:
            actions.remove('u')

        return actions if len(actions) > 0 else None

    def reset(self, episodes, steps, rewords):
        self.start_position = (random.randrange(0, int(self.width / 2)), random.randrange(0, self.height))
        self.refresh_plot()

        if episodes > 0:
            print('Episode ', episodes)
            print('Average Steps ', steps / episodes)
            print('Average Rewords ', rewords / episodes)
            print('')
        return self.start_position


def main():
    width = 10
    height = 7
    stop = (8, 4)

    winds = {5: 1, 6: 2, 7: 2, 8: 1}
    world = WindyGridWorld(width, height, winds, stop)

    '''
    eligibility trace在QLearning上的效果提升没有Sarsa上大，
    因为QLearning在做探索步骤的时候trace会断掉，用于更新的trace没有Sarsa中的长
    '''
    '''
    由于除了终点外，所有状态的奖励都是-1，但Q值的初始值是0，大于-1，
    所有就算e_greedy=0, 算法在一开始也会进行大量的探索
    '''
    sarsa = Sarsa(world, e_greedy=0.0, trace=0.9)
    sarsa.run(1000)

    # 如果q值还没有收敛，即学习次数不够，执行策略有可能陷入死循环
    sarsa.run_policy(100)

    plt.pause(10)


if __name__ == '__main__':
    main()
