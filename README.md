# PhysSim

A simple physics simulator written in C++, just for fun. At the moment it only handles some basic gravitation simulations, but in the future it should include a wide array of physics interactions.

A video of a simulation: https://www.youtube.com/watch?v=22qlvhqwzbs

## Building with vcpkg
In order for the build to succeed, you'll need to have Nvidia's CUDA toolkit installed. On linux there are some additional libraries required, pay attention to the error messages when building to see which libraries are necessary to install.

```bash
> ./vcpgk/bootstrap-vcpkg.sh
> ./vcpkg/vcpkg install
> cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
> cmake --build build
```

## Running with docker
> **Note:** The dockerfiles are currently broken. I'll try to have a fix for them soon.
### Building the simulation docker container
```bash
> docker build . -f docker/simulator.dockerfile --build-arg cuda_arch=86 -t physsim
```
The environment variable `cuda_arch` refers to the identfier for the gpu that cuda will be running on. A list of cuda architectures can be found [here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).

### Starting simulator in docker
Change to the desired output directory in bash, create a `config.yaml` file for the simulation, and then run the following command:
```bash
# Git bash on windows, running in the desired output directory
> docker run -ti --gpus all -v /$(pwd):/simulation physsim
```