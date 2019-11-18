from tkinter import *
import os
import re
import bitstring
import yaml
from PIL import Image, ImageTk
import numpy
import sys

data_regex = re.compile('test_frame_([0-9]+)\.dat')
image_regex = re.compile('test_frame_([0-9]+)\.png')
wDimensions = 1200

class Processor:
    def __init__(self, config):
        self.frame = 0
        self.simulationFrame = 0
        self.files = []
        self.scale = wDimensions / config['dimensions'] * 0.99

        for root, dirs, files in os.walk(os.getcwd()):
            if root == os.getcwd():
                for file in files:
                    if data_regex.match(file):
                        self.files.append(file)

        self.files.sort(key=lambda x: int(data_regex.match(x).group(1)))

    def draw_frame(self):
        if self.frame < len(self.files):
            data = numpy.zeros((wDimensions, wDimensions, 3), dtype=numpy.uint8)
            with open(self.files[self.frame], 'rb') as f:
                self.simulationFrame = int(data_regex.match(self.files[self.frame]).group(1))
                b = bitstring.ConstBitStream(f)
                nParticles = b.read('intle:32')
                for i in range(0, nParticles):
                    data[int(b.read('floatle:32') * self.scale + (wDimensions / 2)),
                         int(b.read('floatle:32') * self.scale + (wDimensions / 2))] = [255, 255, 255]

            image = Image.fromarray(data)
            image.save('test_frame_%d.png' % self.simulationFrame)

            return True
        return False

    def run(self):
        while self.draw_frame():
            print('Frame: %d' % self.simulationFrame)
            self.frame += 1

class Replayer:
    def __init__(self, root, config):
        self.frame = 0
        self.simulationFrame = 0
        self.files = []
        self.scale = wDimensions / config['dimensions'] * 0.99

        self.canvas = Canvas(root, width = wDimensions, height = wDimensions)
        self.canvas.pack()

        for root, dirs, files in os.walk(os.getcwd()):
            if root == os.getcwd():
                for file in files:
                    if image_regex.match(file):
                        self.files.append(file)

        self.files.sort(key=lambda x: int(image_regex.match(x).group(1)))

    def draw_frame(self):
        if self.frame < len(self.files):
            self.simulationFrame = int(image_regex.match(self.files[self.frame]).group(1))
            image = Image.open(self.files[self.frame])
            self.tatras = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=NW, image=self.tatras)

            return True
        return False

    def update(self):
        self.canvas.delete('all')

        if self.draw_frame():
            print('Frame: %d' % self.simulationFrame)

            self.frame += 1
            self.canvas.after(5, self.update)

# Load the config
with open('config.yaml', 'r') as f:
    config = yaml.load(f)

if len(sys.argv) > 1 and sys.argv[1] == '-p':
    processor = Processor(config)
    processor.run()
else:
    # initialize root Window and canvas
    root = Tk()
    root.title("PhysSim")
    root.resizable(False,False)

    replayer = Replayer(root, config)
    replayer.update()

    root.mainloop()