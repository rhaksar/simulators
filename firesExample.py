from collections import defaultdict
import numpy as np

from fires.LatticeForest import LatticeForest

if __name__ == "__main__":
    dimension = 25
    control = defaultdict(lambda: (0, 0))

    alpha = dict()
    alpha_start = 0.1
    alpha_end = 0.4
    for r in range(dimension):
        for c in range(dimension):
            alpha[(r, c)] = alpha_start + (c / (dimension - 1)) * (alpha_end - alpha_start)

    # alpha_dense = np.array([[alpha[(r, c)] for c in range(dimension)]
    #                         for r in range(dimension)])

    sim = LatticeForest(dimension, alpha=alpha)

    for _ in range(20):
        sim.update(control)
        print(sim.dense_state())
        print()

    print()
