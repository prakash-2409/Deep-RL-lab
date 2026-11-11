import numpy as np

class BanditEnvironment:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.num_arms = len(probabilities)

    def pull(self, arm):
        return 1 if np.random.rand() < self.probabilities[arm] else 0

class EpsilonGreedy:
    def __init__(self, num_arms, epsilon=0.1):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm] = new_value

def simulate(env, agent, num_pulls):
    rewards = np.zeros(num_pulls)
    for i in range(num_pulls):
        arm = agent.select_arm()
        reward = env.pull(arm)
        agent.update(arm, reward)
        rewards[i] = reward
    return rewards

probabilities = [0.1, 0.5, 0.8]
env = BanditEnvironment(probabilities)
epsilon = 0.1
num_pulls = 1000

agent = EpsilonGreedy(num_arms=len(probabilities), epsilon=epsilon)
rewards = simulate(env, agent, num_pulls)

print(f"Total reward after {num_pulls} pulls: {np.sum(rewards)}")
print(f"Counts of pulls for each arm: {agent.counts}")
print(f"Estimated values for each arm: {agent.values}")
