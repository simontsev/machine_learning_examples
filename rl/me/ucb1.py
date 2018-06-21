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
        self.bandits = [
            BanditStat(Bandit(0.1)), 
            BanditStat(Bandit(0.2)),
            BanditStat(Bandit(0.3))]

    def play_bandit(self, num):
        success = self.bandits[num].bandit.play()
        self.bandits[num].update(success)
        return success


class BanditStat(object):

    def __init__(self, bandit):
        self.bandit = bandit
        self.attempts = 0
        self.wins = 0
        self.mean = 0

    def update(self, success):
        self.attempts += 1
        if success == 1:
            self.wins += 1
        self.mean = self.wins / float(self.attempts)


class Agent(object):

    def __init__(self, environment):
        self.environment = environment

    def play(self, number_of_attempts):

        bandit_num = np.argmax(
            [b.mean + 
                np.sqrt(
                    2.0*np.log(number_of_attempts)/(b.attempts if b.attempts > 0 else 0.01)
                ) 
            for b in self.environment.bandits]
        )

        return self.environment.play_bandit(bandit_num)


def main():

    num_plays = 100000

    env = Environment()
    agent = Agent(env)

    data = np.empty(num_plays)
    for i in range(num_plays):
        data[i] = agent.play(i+1)
            
    
    print("Agent played {} times".format(num_plays))
    print("Bandit 1 wins: {}".format(env.bandits[0].wins))
    print("Bandit 2 wins: {}".format(env.bandits[1].wins))
    print("Bandit 3 wins: {}".format(env.bandits[2].wins))
    total = np.sum(data)
    print("Total wins: {}".format(total))

    cumulative_average = np.cumsum(data) / (np.arange(num_plays) + 1)
    plt.plot(cumulative_average, label="ucb1")
    plt.legend()
    plt.xscale('log')
    plt.show()

if __name__ == "__main__":
    main()
