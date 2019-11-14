#include "Core.h"

#include <chrono>
#include <thread>
#include <fstream>

Core::Core(float width, float height, bool interParticleGravity) {
    this->interParticleGravity = interParticleGravity;
	this->particles = nullptr;
	this->nParticles = 0;

	this->massiveParticles = nullptr;
	this->nMassiveParticles = 0;
	
	this->timeStep = 1;
	this->stepsPerFrame = 1;
	this->framesPerSecond = 100;

	this->stepCount = 0;
}
void Core::setGravity(Vector gravity) {
    this->gravity = gravity;
}

void Core::setTimeStep(float timeStep) {
	this->timeStep = timeStep;
}

void Core::setStepsPerFrame(float stepsPerFrame) {
	this->stepsPerFrame = stepsPerFrame;
}

void Core::setFramesPerSecond(float framesPerSecond) {
	this->framesPerSecond = framesPerSecond;
}

void Core::addParticles(Particle* particles, int nParticles) {
	this->particles = particles;
	this->nParticles = nParticles;
}

void Core::addMassiveParticles(Particle* massiveParticles, int nMassiveParticles) {
	this->massiveParticles = massiveParticles;
	this->nMassiveParticles = nMassiveParticles;
}

void Core::run() {
    while (true) {
        // Call cuda to think here
		// Launch a kernel on the GPU with one thread for each element.
		cudaDeviceSynchronize();

		++stepCount;
    }


	cudaFree(particles);
	cudaFree(massiveParticles);
	cudaDeviceReset();
}
