# PhysSim

A simple physics simulator written in C++, just for fun. At the moment it only handles some basic gravitation simulations, but in the future it should include a wide array of physics interactions.

A video of a simulation: https://www.youtube.com/watch?v=22qlvhqwzbs


### Running simulator
#### Building the simulation docker container
```bash
> docker build . -f docker/simulator.dockerfile --build-arg cuda_arch=86 -t physsim
```
The environment variable `cuda_arch` refers to the identfier for the gpu that cuda will be running on. A list of cuda architectures can be found [here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).

### Running
#### Starting simulator in docker
Change to the desired output directory in bash, create a `config.yaml` file for the simulation, and then run the following command:
```bash
# Git bash on windows, running in the desired output directory
> docker run -ti --gpus all -v /$(pwd):/simulation physsim
```

### Building on Windows

```bash
# Build simulator and replayer on windows
> mkdir build && cd build
> conan install .. -s build_type=Release --build missing
> cmake .. && cmake --build . --config Release
```

Alternatively, there's a docker container available for convenience that can be built with docker/build-windows.dockerfile:
```bash
> docker build -f docker/build/windows.dockerfile -t build-physsim .
# The command to run the container with the local project directory mounted (using git bash on windows)
> docker run -ti -v `pwd -W`:c:/project build-physsim:latest
```