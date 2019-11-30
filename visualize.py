from tkinter import *
import os
import re
import bitstring
import yaml
from PIL import Image, ImageTk
import numpy
import sys
import time
import struct
import io
import aggdraw
import threading

position_folder = 'position'
image_folder = 'image'

data_regex = re.compile('position-([0-9]+)\.dat')
image_regex = re.compile('position-([0-9]+)\.png')

wDimensions = 1200
framesPerWrite = 1000
nThreads = 8

def getNParticles(config):
    nParticles = 0
    for particle in config['particles']:
        if 'n' in particle:
            nParticles += particle['n']
        else:
            nParticles += 1

    return nParticles

class Processor(threading.Thread):
    def __init__(self, config, files, threadId):
        super().__init__()
        self.frame = 0
        self.simulationFrame = 0
        self.files = files
        self.scale = wDimensions / config['dimensions'] * 0.99
        self.nParticles = getNParticles(config)
        self.threadId = threadId
        self.alive = True

    def start_timer(self):
        self.timer = time.time()

    def log_timer(self, message):
        print('[thread %d] %s: %f' % (self.threadId, message, time.time() - self.timer))

    def draw_frame(self):
        if self.frame < len(self.files):
            self.start_timer()
            data = numpy.full((wDimensions, wDimensions, 3), [255, 0, 0], dtype=numpy.uint8)
            with open(os.path.join(position_folder, self.files[self.frame]), 'rb') as f:
                self.simulationFrame = int(data_regex.match(self.files[self.frame]).group(1))
                b = bitstring.ConstBitStream(f.read())

            image = Image.fromarray(data)
            d = aggdraw.Draw(image)
            p = aggdraw.Pen("black", 0.3, 240)
            for i in range(0, self.nParticles):
                x = int(b.read('floatle:32') * self.scale + (wDimensions / 2))
                y = int(b.read('floatle:32') * self.scale + (wDimensions / 2))
                d.ellipse((x - 1, y - 1, x + 1, y + 1), p)
            d.flush()
            
            image.save(os.path.join(image_folder, 'position-%d.png' % self.simulationFrame))
            self.log_timer('time')

            return True
        return False

    def run(self):
        while self.alive and self.draw_frame():
            print('Frame: %d' % self.simulationFrame)
            self.frame += framesPerWrite

    def kill(self):
        self.alive = False

class Replayer:
    def __init__(self, root, config, simulationFrame = 0):
        self.frame = 0
        self.simulationFrame = 0
        self.files = []
        self.scale = wDimensions / config['dimensions'] * 0.99

        self.canvas = Canvas(root, width = wDimensions, height = wDimensions)
        self.canvas.pack()

        for root, dirs, files in os.walk(os.path.join(os.getcwd(), image_folder)):
            if root == os.path.join(os.getcwd(), image_folder):
                for file in files:
                    if image_regex.match(file):
                        self.files.append(file)

        self.files = [x for x in self.files if int(image_regex.match(x).group(1)) > simulationFrame]
        self.files.sort(key=lambda x: int(image_regex.match(x).group(1)))

    def draw_frame(self):
        if self.frame < len(self.files):
            if os.path.exists(os.path.join(image_folder, self.files[self.frame])):
                self.simulationFrame = int(image_regex.match(self.files[self.frame]).group(1))
                image = Image.open(os.path.join(image_folder, self.files[self.frame]))
                self.tatras = ImageTk.PhotoImage(image)
                self.canvas.create_image(0, 0, anchor=NW, image=self.tatras)

            return True
        return False

    def update(self):
        self.canvas.delete('all')

        if self.draw_frame():
            print('Frame: %d' % self.simulationFrame)

            self.frame += 1
            self.canvas.after(10, self.update)
        else:
            sys.exit(0)


# Load the config
with open('config.yaml', 'r') as f:
    config = yaml.load(f)

if not os.path.exists(image_folder):
    os.makedirs(image_folder)

if len(sys.argv) > 1 and sys.argv[1] == '-p':
    currentFrame = 0
    simFiles = []
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), position_folder)):
        if root == os.path.join(os.getcwd(), position_folder):
            for file in files:
                if data_regex.match(file):
                    simFiles.append(file)
                if image_regex.match(file):
                    frame = int(image_regex.match(file).group(1))
                    if frame > currentFrame:
                        currentFrame = frame

    simFiles = [x for x in simFiles if int(data_regex.match(x).group(1)) > currentFrame]
    simFiles.sort(key=lambda x: int(data_regex.match(x).group(1)))

    fileLists = []
    for i in range(0, nThreads):
        fileLists.append(list())

    for i in range(0, len(simFiles)):
        fileLists[i % nThreads].append(simFiles[i])

    processors = []
    for i in range(0, nThreads):
        processor = Processor(config, fileLists[i], i)
        processors.append(processor)
        processor.start()

    try:
        while True:
            done = True
            for processor in processors:
                if processor.frame < len(processor.files):
                    done = False

            if(done):
                sys.exit(0)
    except KeyboardInterrupt:
        print('Shutting down')

    for processor in processors:
        processor.kill()
else:
    # initialize root Window and canvas
    root = Tk()
    root.title("PhysSim")
    root.resizable(False,False)

    startFrame = 0
    if len(sys.argv) > 1:
        startFrame = int(sys.argv[1])

    replayer = Replayer(root, config, startFrame)
    replayer.update()

    root.mainloop()