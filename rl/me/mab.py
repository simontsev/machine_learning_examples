import numpy as np


class BanditBernoulli(object):

    def __init__(self, probability_of_1):
        self.p_of_1 = probability_of_1

    def play(self):
        if np.random.uniform() < self.p_of_1:
            return 1
        return 0


class Bandit(object):

    def __init__(self, mean_actual):
        self.mean_actual = mean_actual

    def play(self):
        return np.random.randn() + self.mean_actual


class Environment(object):

    def __init__(self, true_means, use_bernoulli=False):
        if use_bernoulli:
            self.bandits = [BanditBernoulli(m) for m in true_means]
        else:
            self.bandits = [Bandit(m) for m in true_means]

    def play_bandit(self, num):
        return self.bandits[num].play()


def main():

    env = Environment([0.1, 0.2, 0.3])
    env_b = Environment([0.1, 0.2, 0.3], use_bernoulli=True)
    num_plays = 100000
    data = np.empty(num_plays)
    data_b = np.empty(num_plays)
    for i in range(num_plays):
        data[i] = env.play_bandit(2)
        data_b[i] = env_b.play_bandit(2)

    print("Total Reward: {}".format(np.sum(data)))
    print("Total Reward Bernoulli: {}".format(np.sum(data_b)))


if __name__ == "__main__":
    main()
