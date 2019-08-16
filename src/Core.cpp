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

void Core::calcMassiveParticles(float timeElapsed, int step)
{
	for (int i = 0; i < nMassiveParticles; i++) {
		Vector* position = &massiveParticles[i].position[step % 2];
		Vector* lastPosition = &massiveParticles[i].position[(step + 1) % 2];

		v3 force = { 0 };

		// Add the force from the bounds
		float radiusFromCenter = sqrt(pow(lastPosition->x - center.x, 2.0) + pow(lastPosition->y - center.y, 2.0) + pow(lastPosition->z - center.z, 2.0));
		v3 direction = { center.x - lastPosition->x, center.y - lastPosition->y, center.z - lastPosition->z };
		float magnitude = pow(radiusFromCenter, 2.0f) * massiveParticles[i].mass * FORCE_CONST;
		direction.x /= radiusFromCenter;
		direction.y /= radiusFromCenter;
		direction.z /= radiusFromCenter;

		force.x += direction.x * magnitude;
		force.y += direction.y * magnitude;
		force.z += direction.z * magnitude;

		// Add the force from the particles in the medium
		for (int o = 0; o < nParticles; o++) {
			Vector* otherPosition = &particles[o].position[(step + 1) % 2];
		
			float dist = sqrt(pow(otherPosition->y - lastPosition->y, 2.0) + pow(otherPosition->x - lastPosition->x, 2.0) + pow(otherPosition->z - lastPosition->z, 2.0));
			if (dist > 0.0) {
				// XXX Review this, should be dist^2
				float forceMagnitude = (G * (float)massiveParticles[i].mass * (float)particles[o].mass) / pow(dist, 2.0);
		
				v3 v = { lastPosition->x - otherPosition->x, lastPosition->y - otherPosition->y, lastPosition->z - otherPosition->z };
				float vectorMagnitude = sqrt(pow(v.x, 2.0) + pow(v.y, 2.0) + pow(v.z, 2.0));
				v.x /= vectorMagnitude;
				v.y /= vectorMagnitude;
				v.z /= vectorMagnitude;
		
				force.x += v.x * forceMagnitude;
				force.y += v.y * forceMagnitude;
				force.z += v.z * forceMagnitude;
			}
		}

		 v3 acceleration = { force.x / massiveParticles[i].mass, force.y / massiveParticles[i].mass, force.z / massiveParticles[i].mass };
		 
		 position->x = lastPosition->x + (massiveParticles[i].velocity.x * timeElapsed) + (0.5 * acceleration.x * pow(timeElapsed, 2.0));
		 position->y = lastPosition->y + (massiveParticles[i].velocity.y * timeElapsed) + (0.5 * acceleration.y * pow(timeElapsed, 2.0));
		 position->z = lastPosition->z + (massiveParticles[i].velocity.z * timeElapsed) + (0.5 * acceleration.z * pow(timeElapsed, 2.0));
		 
		 massiveParticles[i].velocity.x = massiveParticles[i].velocity.x + (acceleration.x * timeElapsed);
		 massiveParticles[i].velocity.y = massiveParticles[i].velocity.y + (acceleration.y * timeElapsed);
		 massiveParticles[i].velocity.z = massiveParticles[i].velocity.z + (acceleration.z * timeElapsed);
	}
}

void Core::run() {
    while (true) {
		if (stepCount % stepsPerFrame == 0)
			ui->tick(particles, nParticles, massiveParticles, nMassiveParticles, stepCount + 1);

        if (ui->shouldClose())
            break;

		if (stepCount % 100 == 0)
		{
			dumpToDisk("medium_22000_r2_2d_5_" + std::to_string(stepCount) + ".dmp");
		}

        // Call cuda to think here
		// Launch a kernel on the GPU with one thread for each element.
		doWorkHypothetical(particles, nParticles, massiveParticles, nMassiveParticles, timeStep, center, radius, stepCount);
		cudaDeviceSynchronize();
// 		calcMassiveParticles(timeStep, stepCount);

		++stepCount;

		if (stepCount == 1)
		{
			Particle* massiveParticles;
			const int numMassiveParticles = 1;

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
				massiveParticles[i].setInitialPosition(Vector(0, 0, 0));

				massiveParticles[i].setMass(500000.0);
				massiveParticles[i].setColor(1, 1, 1);
				massiveParticles[i].setRadius(30);
			}

			addMassiveParticles(massiveParticles, numMassiveParticles);
		}
    }

    ui->terminate();

    delete ui;

	cudaFree(particles);
	cudaFree(massiveParticles);
	cudaDeviceReset();
}
