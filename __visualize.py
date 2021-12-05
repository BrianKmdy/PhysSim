import bitstring
import matplotlib.pyplot as plt
import math

nParticles = 0
particles = []

def dist(p1, p2):
    return math.sqrt(math.pow(p2['x'] - p1['x'], 2) + math.pow(p2['y'] - p1['y'], 2) + math.pow(p2['z'] - p1['z'], 2))

def graphParticles(particles):
    x = [p['x'] for p in particles]
    y = [p['y'] for p in particles]

    plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.scatter(x, y, s=1)
    plt.show()

with open('PhysSim/test_frame_700.dat', 'rb') as f:
    b = bitstring.ConstBitStream(f)

    nParticles = b.read('intle:32')

    for i in range(0, nParticles):
        particle = {'x': b.read('floatle:32'), 'y': b.read('floatle:32')}
        particles.append(particle)

graphParticles(particles)