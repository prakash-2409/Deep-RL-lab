import numpy as np

actual_prob = [0.1, 0.7, 0.5]
succ_fail = [[0, 0], [0, 0], [0, 0]]

for trial in range(101):
    samples = [np.random.beta(s + 1, f + 1) for s, f in succ_fail]
    best_arm = np.argmax(samples)

    if np.random.uniform() < actual_prob[best_arm]:
        succ_fail[best_arm][0] += 1
    else:
        succ_fail[best_arm][1] += 1

    if trial % 10 == 0:
        print(f"trial: {trial}\tsucc_fail: {succ_fail}")
