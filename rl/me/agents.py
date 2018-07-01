import numpy as np
from explore_exploit import Ucb1BanditStat, BayesianBanditStat

class Agent(object):

    def __init__(self, environment):
        self.environment = environment

    def play(self, number_of_attempts):
        pass


class Ucb1Agent(Agent):

    def __init__(self, environment, verbose=False):
        Agent.__init__(self, environment)
        self.bandit_stats = [Ucb1BanditStat() for b in environment.bandits]

        self.verbose = verbose

    def play(self, number_of_attempts):

        bandit_num = np.argmax(
            [b.predicted_mean + 
                np.sqrt(
                    2.0*np.log(number_of_attempts)/(b.attempts if b.attempts > 0 else 0.01)
                ) 
            for b in self.bandit_stats]
        )

        if self.verbose:
            print("Ucb1Agent: Playing bandit #{}".format(bandit_num))
        
        x = self.environment.play_bandit(bandit_num)
        self.bandit_stats[bandit_num].update(x, number_of_attempts)

        if self.verbose:
            print("Ucb1Agent: Bandit #{} predicted mean {}".format(bandit_num, self.bandit_stats[bandit_num].predicted_mean))

        return x


class BayesianAgent(Agent):

    def __init__(self, environment):
        Agent.__init__(self, environment)
        self.bandit_stats = [BayesianBanditStat() for b in environment.bandits]

    def play(self, number_of_attempts):

        bandit_num = np.argmax([b.sample for b in self.bandit_stats])

        x = self.environment.play_bandit(bandit_num)
        self.bandit_stats[bandit_num].update(x, number_of_attempts)
        return x

