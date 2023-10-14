import matplotlib.pyplot as plt
import numpy as np
from math import sin,exp

def makeArray(n, m):
    a = []
    time = np.linspace(0,m-1,m)
    for i in range(n):
        x = [sin((i-t*0.01)*10) for t in range(m)]
        a.append(x)
    return a, time

def findSpikes(state, time):
    result =  []
    for x in state:
        indices = []
        for xx in x:
            if (xx > 0.95):
                indices.append(x.index(xx))
        result.append([time[i] for i in indices])
    return result

x,t = makeArray(10,1000)
spikes = findSpikes(x,t)
fig, [ax1,ax2] = plt.subplots(1,2)
fig.suptitle("Grafici dell'attività neuronale")
ax1.plot(t, x[4])
ax1.set_xlabel("Time")
ax1.set_ylabel("Voltage")
ax1.set_title("Attività del quinto neurone")
ax2.eventplot(spikes, orientation='vertical', colors=[[i*0.1+0.1,0,0.5] for i in range(10)])
ax2.set_xlabel("Neurons")
ax2.set_ylabel("Time")
ax2.set_title("Raster plot")
plt.show()
