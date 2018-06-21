import numpy as np
import matplotlib.pyplot as plt

class Bandit(object):

    def __init__(self, probability_of_1):
        self.p_of_1 = probability_of_1

    def play(self):
        if np.random.uniform() < self.p_of_1:
            return 1
        return 0


class Environment(object):

    def __init__(self):
        self.bandit_1 = Bandit(0.1)
        self.bandit_2 = Bandit(0.2)
        self.bandit_3 = Bandit(0.3)

    def play_bandit(self, num):
        if num == 1:
            return self.bandit_1.play()
        if num == 2:
            return self.bandit_2.play()
        return self.bandit_3.play()


class Agent(object):

    def __init__(self, environment, epsilon, best_bandit_num=1):
        self.environment = environment
        self.epsilon = epsilon
        self.best_bandit_num = best_bandit_num
        self.result = np.array([0,0,0])
        self.attempts = np.array([0.0001,0.0001,0.0001])

    def play(self):

        if np.random.uniform() < self.epsilon:
            bandit_num = np.random.randint(1,4,size=1)[0]
        else:
            bandit_num = self.best_bandit_num

        success = self.environment.play_bandit(bandit_num)
        self.attempts[bandit_num-1] += 1
        if success == 1:
            self.result[bandit_num-1] += 1
            self.best_bandit_num = \
                np.argmax(self.result/self.attempts) + 1
        
        return success


def main():

    attempts = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    num_plays = 100000
    best_total = 0
    best_epsilon = 0

    for attempt in attempts:
        epsilon = attempt
        env = Environment()
        agent = Agent(env, epsilon)

        data = np.empty(num_plays)
        for i in range(num_plays):
            data[i] = agent.play()
             
        
        print("Agent played {} times (epsilon={})".format(num_plays, epsilon))
        print("Bandit 1 wins: {}".format(agent.result[0]))
        print("Bandit 2 wins: {}".format(agent.result[1]))
        print("Bandit 3 wins: {}".format(agent.result[2]))
        total = np.sum(agent.result)
        print("Total wins: {}".format(total))

        cumulative_average = np.cumsum(data) / (np.arange(num_plays) + 1)
        if total > best_total:
            best_total = total
            best_epsilon = epsilon
        plt.plot(cumulative_average, label="epsilon {}".format(attempt))

    print("Best epsilon {} with total wins {}".format(best_epsilon, best_total))
    plt.legend()
    plt.xscale('log')
    plt.show()

if __name__ == "__main__":
    main()
