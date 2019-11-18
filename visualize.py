from tkinter import *
import os
import re
import bitstring
import yaml
from PIL import Image, ImageTk
import numpy

file_regex = re.compile('test_frame_([0-9]+)\.dat')
wDimensions = 1200

class Processor:
    def __init__(self, root, config):
        self.frame = 0
        self.particles = []
        self.files = []
        self.radius = 1
        self.scale = wDimensions / config['dimensions'] * 0.99

        self.canvas = Canvas(root, width = wDimensions, height = wDimensions)
        self.canvas.pack()

        for root, dirs, files in os.walk(os.getcwd()):
            if root == os.getcwd():
                for file in files:
                    if file_regex.match(file):
                        self.files.append(file)

        self.files.sort(key=lambda x: int(file_regex.match(x).group(1)))

    def draw_frame(self):
        if self.frame < len(self.files):
            data = numpy.zeros((wDimensions, wDimensions, 3), dtype=numpy.uint8)
            

            with open(self.files[self.frame], 'rb') as f:
                b = bitstring.ConstBitStream(f)
                nParticles = b.read('intle:32')
                for i in range(0, nParticles):
                    data[int(b.read('floatle:32') * self.scale + (wDimensions / 2)),
                         int(b.read('floatle:32') * self.scale + (wDimensions / 2))] = [255, 255, 255]

            image = Image.fromarray(data)
            self.tatras = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=NW, image=self.tatras)

            return True

        return False

    def update(self):
        self.canvas.delete('all')
        self.particles.clear()

        if self.draw_frame():

            self.frame += 1
            self.canvas.after(5, self.update)

class Replayer:
    def __init__(self):
        pass

# Load the config
with open('config.yaml', 'r') as f:
    config = yaml.load(f)

# initialize root Window and canvas
root = Tk()
root.title("PhysSim")
root.resizable(False,False)

# create two ball objects and animate them
simulation = Processor(root, config)
simulation.update()

root.mainloop()