#include "Core.h"

Core::Core() :
	instance(new Instance())
{
}

Core::Core(Instance* instance):
	instance(instance)
{
}

Core::~Core()
{
	if (instance) {
		if (instance->particles)
			delete[] instance->particles;
		if (instance->boxes)
			delete[] instance->boxes;

		delete instance;
	}
}

Instance* Core::getInstance() {
	return instance;
}

void Core::run() {
	simulate(instance);
    // while (true) {
    //     // Call cuda to think here
	// 	// Launch a kernel on the GPU with one thread for each element.
	// 	cudaDeviceSynchronize();
    // }


	// cudaFree(particles);
	// cudaFree(massiveParticles);
	// cudaDeviceReset();
}
