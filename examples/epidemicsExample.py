from simulators.epidemics.WestAfrica import WestAfrica


if __name__ == '__main__':
    # specify initial outbreak location
    outbreak = {('guinea', 'gueckedou'): 1}

    # instantiate simulation object
    sim = WestAfrica(outbreak)

    # step forward
    for _ in range(75):
        sim.update()

    # print state
    state = sim.dense_state()
    for name in state:
        print('name: %s, state: %d' % (name, state[name]))
