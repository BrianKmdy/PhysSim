#include "Core.h"

#include <chrono>
#include <thread>

Core::Core(double width, double height, bool interParticleGravity) {
    this->interParticleGravity = interParticleGravity;
	this->entities = nullptr;
	this->nParticles = 0;

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

void Core::setOutput(int output) {
    this->output = output;
    ui->setOutput(output);
}

void Core::setFps(double fps) {
    this->fps = fps;
}

void Core::addEntities(Particle* particles, int nParticles)
{
	this->entities = particles;
	this->nParticles = nParticles;
}

GUI *Core::getGUI() {
    return ui;
}

void Core::run() {
	std::chrono::nanoseconds startTime =
		std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());

    double timeElapsed = 0.0;

    while (true) {
        ui->tick(entities, nParticles);

        if (ui->shouldClose())
            break;

		std::chrono::nanoseconds currentTime =
			std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());

		double timeElapsed = (output == GUI::OUTPUT_TO_SCREEN) ? (currentTime - startTime).count() / 1000000000.0f * rate : (1.0 / fps) * rate;

        // Call cuda to think here
		// Launch a kernel on the GPU with one thread for each element.
		doWork(entities, nParticles, timeElapsed);
		cudaDeviceSynchronize();

		startTime = currentTime;
    }

	ui->tick(entities, nParticles);
    ui->terminate();

    delete ui;

	cudaFree(entities);
	cudaDeviceReset();
}
