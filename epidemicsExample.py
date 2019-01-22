from collections import defaultdict
import numpy as np
import pickle

from epidemics.WestAfrica import WestAfrica


if __name__ == '__main__':
    # import graph data
    file = open('west_africa_graph.pkl', 'rb')
    graph = pickle.load(file)
    file.close()

    # specify initial outbreak location
    outbreak = {('guinea', 'gueckedou'): 1}

    # instantiate simulation object
    sim = WestAfrica(graph, outbreak)

    # step forward
    for _ in range(75):
        sim.update()

    # print state
    state = sim.dense_state()
    for name in state:
        print('name: %s, state: %d' % (name, state[name]))
