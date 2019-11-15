#include "Core.h"


Core::Core(int dimensions, int divisions, int nParticles, Particle* particles):
	dimensions(dimensions),
	divisions(divisions),
	instance()
{
}

Core::~Core()
{
}

void Core::run() {
    // while (true) {
    //     // Call cuda to think here
	// 	// Launch a kernel on the GPU with one thread for each element.
	// 	cudaDeviceSynchronize();
    // }


	// cudaFree(particles);
	// cudaFree(massiveParticles);
	// cudaDeviceReset();
}
