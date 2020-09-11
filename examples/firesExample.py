from simulators.fires.LatticeForest import LatticeForest

if __name__ == "__main__":
    # forest lattice size
    dimension = 10

    # create a non-uniform fire propagation parameter to model the effects of wind
    alpha = dict()
    alpha_start = 0.1
    alpha_end = 0.4
    for r in range(dimension):
        for c in range(dimension):
            alpha[(r, c)] = alpha_start + (c/(dimension-1))*(alpha_end-alpha_start)

    # instantiate simulator
    sim = LatticeForest(dimension, alpha=alpha)

    # step forward and print state
    for _ in range(5):
        # sim.update(control)
        sim.update()
        print('iteration %d' % sim.iter)
        print(sim.dense_state())
        print()
