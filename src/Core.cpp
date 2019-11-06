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

    this->ui = new GUI(width, height);
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

void Core::setOutput(int output) {
    this->output = output;
    ui->setOutput(output, framesPerSecond);
}

void Core::addParticles(Particle* particles, int nParticles) {
	this->particles = particles;
	this->nParticles = nParticles;
}

void Core::addMassiveParticles(Particle* massiveParticles, int nMassiveParticles) {
	this->massiveParticles = massiveParticles;
	this->nMassiveParticles = nMassiveParticles;
}

GUI *Core::getGUI() {
    return ui;
}

#include <iostream>
void Core::dumpToDisk(std::string filename, bool full)
{
	std::ofstream f;
	f.open(filename, std::ofstream::out | std::ios::binary);
	f.write((char*) &nParticles, sizeof(int));
	f.write((char*) &nMassiveParticles, sizeof(int));

	for (int i = 0; i < nParticles; ++i)
	{
		f.write((char*) &particles[i].position[stepCount % 2], sizeof(Vector));
	}

	for (int i = 0; i < nMassiveParticles; ++i)
	{
		f.write((char*) &massiveParticles[i].position[stepCount % 2], sizeof(Vector));
	}
}

void Core::loadFromDisk(std::string filename, bool full)
{

}

void Core::run() {
    while (true) {
		if (stepCount % stepsPerFrame == 0)
			ui->tick(particles, nParticles, massiveParticles, nMassiveParticles, stepCount + 1);

        if (ui->shouldClose())
            break;

		if (stepCount % 100 == 0)
		{
			dumpToDisk("medium_100000_2d_bounded_5_" + std::to_string(stepCount) + ".dmp");
		}

        // Call cuda to think here
		// Launch a kernel on the GPU with one thread for each element.
		doWorkHypothetical(particles, nParticles, massiveParticles, nMassiveParticles, timeStep, center, radius, stepCount);
		cudaDeviceSynchronize();

		++stepCount;

		if (stepCount == 1)
		{
			Particle* massiveParticles;
			const int numMassiveParticles = 2;

			// Initialize the massive particles
			if (numMassiveParticles > 0)
			{
				cudaError cudaStatus;
				cudaStatus = cudaMallocManaged(&massiveParticles, numMassiveParticles * sizeof(Particle));
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMalloc failed!");
					return;
				}
			}

			massiveParticles[0].setInitialPosition(Vector(-750, 0, 0));
			massiveParticles[0].setMass(50000.0);
			massiveParticles[0].setColor(1, 1, 1);
			massiveParticles[0].setRadius(30);

			massiveParticles[1].setInitialPosition(Vector(750, 0, 0));
			massiveParticles[1].setMass(50000.0);
			massiveParticles[1].setColor(1, 1, 1);
			massiveParticles[1].setRadius(30);

//			for (int i = 0; i < numMassiveParticles; i++) {
//				massiveParticles[i].setInitialPosition(Vector(0, 0, 0));
//
//				massiveParticles[i].setMass(100000.0);
//				massiveParticles[i].setColor(1, 1, 1);
//				massiveParticles[i].setRadius(30);
//			}

			addMassiveParticles(massiveParticles, numMassiveParticles);
		}
    }

    ui->terminate();

    delete ui;

	cudaFree(particles);
	cudaFree(massiveParticles);
	cudaDeviceReset();
}
