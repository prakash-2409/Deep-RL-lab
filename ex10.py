import numpy as np

class UCB:
    def __init__(self, n_arms, c=2):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        total_counts = np.sum(self.counts)
        if 0 in self.counts:
            return np.argmin(self.counts)
        ucb_values = self.values + self.c * np.sqrt(np.log(total_counts) / (self.counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / n

def main():
    n_arms, n_trials = 10, 1000
    true_means = np.random.rand(n_arms)
    ucb = UCB(n_arms)
    rewards = np.zeros(n_trials)

    for t in range(n_trials):
        arm = ucb.select_arm()
        reward = np.random.randn() + true_means[arm]
        ucb.update(arm, reward)
        rewards[t] = reward

    print("Total reward:", np.sum(rewards))

if __name__ == "__main__":
    main()
