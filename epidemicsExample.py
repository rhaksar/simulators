from collections import defaultdict
import numpy as np
import pickle

from epidemics.WestAfrica import WestAfrica

handle = open('west_africa_graph.pkl', 'rb')
graph = pickle.load(handle)
handle.close()

control = defaultdict(lambda: (0, 0))
outbreak = {('guinea', 'gueckedou'): 1}

sim = WestAfrica(graph, outbreak)

for _ in range(50):
    sim.update(control)

values = []
for name in sim.group.keys():
    print(name, sim.group[name].state, sim.counter[name])
    values.append(sim.counter[name])
print(np.amin(values))
print(np.mean(values))
print(np.amax(values))
print(np.std(values))

print()
