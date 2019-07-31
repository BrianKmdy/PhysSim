#include "Core.h"

#include <chrono>
#include <thread>

Core::Core(double width, double height, bool interParticleGravity) {
    this->interParticleGravity = interParticleGravity;
	this->entities = nullptr;
	this->nParticles = 0;

	this->massiveParticles = nullptr;
	this->nMassiveParticles = 0;
	
	this->timeStep = 1;
	this->stepsPerFrame = 1;
	this->framesPerSecond = 100;

	this->stepCount = 0;

    this->ui = new GUI(width, height);
}

void Core::setRate(double rate) {
    this->rate = rate;
}

void Core::setG(double G) {
    this->G = G;
}

void Core::setGravity(Vector gravity) {
    this->gravity = gravity;
}

void Core::setTimeStep(double timeStep) {
	this->timeStep = timeStep;
}

void Core::setStepsPerFrame(double stepsPerFrame) {
	this->stepsPerFrame = stepsPerFrame;
}

void Core::setFramesPerSecond(double framesPerSecond) {
	this->framesPerSecond = framesPerSecond;
}

void Core::setOutput(int output) {
    this->output = output;
    ui->setOutput(output, framesPerSecond);
}

void Core::addEntities(Particle* particles, int nParticles) {
	this->entities = particles;
	this->nParticles = nParticles;
}

void Core::addMassiveParticles(Particle* massiveParticles, int nMassiveParticles) {
	this->massiveParticles = massiveParticles;
	this->nMassiveParticles = nMassiveParticles;
}

GUI *Core::getGUI() {
    return ui;
}

void Core::run() {
    while (true) {
		if (stepCount % stepsPerFrame == 0)
			ui->tick(entities, nParticles, massiveParticles, nMassiveParticles);

        if (ui->shouldClose())
            break;

        // Call cuda to think here
		// Launch a kernel on the GPU with one thread for each element.
		doWorkHypothetical(entities, nParticles, massiveParticles, nMassiveParticles, timeStep, center, radius);
		cudaDeviceSynchronize();

		++stepCount;

		if (stepCount == 300)
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

			for (int i = 0; i < numMassiveParticles; i++) {
				massiveParticles[i].setPosition(Vector(static_cast<double>(rand() % 200) - 100, static_cast<double>(rand() % 200) - 100, static_cast<double>(rand() % 200) - 100));

				massiveParticles[i].setMass(5.0);
				massiveParticles[i].setColor(1, 1, 1);
				massiveParticles[i].setRadius(30);
			}

			addMassiveParticles(massiveParticles, numMassiveParticles);
		}
    }

    ui->terminate();

    delete ui;

	cudaFree(entities);
	cudaDeviceReset();
}
