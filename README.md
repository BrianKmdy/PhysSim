# PhysSim

A simple physics simulator written in C++, just for fun. At the moment it only handles some basic gravitation simulations, but in the future it should include a wide array of physics interactions.

A video of a simulation: https://www.youtube.com/watch?v=22qlvhqwzbs

## Building with vcpkg
```bash
> ./vcpgk/bootstrap-vcpkg.sh
> ./vcpkg/vcpkg install
> cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
> cmake --build build
```

## Running simulator
### Building the simulation docker container
```bash
> docker build . -f docker/simulator.dockerfile --build-arg cuda_arch=86 -t physsim
```
The environment variable `cuda_arch` refers to the identfier for the gpu that cuda will be running on. A list of cuda architectures can be found [here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).

## Running
### Starting simulator in docker
Change to the desired output directory in bash, create a `config.yaml` file for the simulation, and then run the following command:
```bash
# Git bash on windows, running in the desired output directory
> docker run -ti --gpus all -v /$(pwd):/simulation physsim
```