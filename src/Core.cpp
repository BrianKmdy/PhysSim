#include <map>
#include <fstream>

#include "spdlog/spdlog.h"

#include "Core.h"
#include "Paths.h"
#include "Types.h"

Core::Core():
	Core(nullptr, nullptr, nullptr)
{
}

Core::Core(Instance* instance, Particle* particles, Box* boxes):
	alive(true),
	instance(instance),
	particles(particles),
	boxes(boxes),
	frame(0),
	framesPerPosition(1),
	framesPerState(100),
	kernel(Kernel::unknown),
	kernelName("not set"),
	startTime(std::chrono::milliseconds(0)),
	frameTime(std::chrono::milliseconds(0))
{
}

Core::~Core()
{
	if (this->instance)
		delete[] instance;
	if (this->particles)
		delete[] particles;
	if (this->boxes)
		delete[] boxes;
}

void Core::verifyConfiguration()
{
	if (!instance)
		throw std::exception("No instance set");

	spdlog::info("---");
	spdlog::info("Config settings");

	std::map<std::string, std::string> values;
	std::map<std::string, std::vector<std::string>> errors;

	// Check the kernel
	values["kernel"] = kernelName;
	if (kernel == Kernel::unknown)
		errors["kernel"].push_back("invalid kernel");

	// Check the dimensions
	values["dimensions"] = std::to_string(instance->dimensions);
	if (instance->dimensions & (instance->dimensions - 1))
		errors["dimensions"].push_back("must be a power of 2");
	if (instance->dimensions <= 0)
		errors["dimensions"].push_back("must be greater than 0");

	// Check the divisions
	values["divisions"] = std::to_string(instance->divisions);
	if (instance->divisions & (instance->divisions - 1))
		errors["divisions"].push_back("must be a power of 2");
	if (instance->divisions <= 0)
		errors["divisions"].push_back("must be greater than 0");
	if (instance->divisions >= instance->dimensions)
		errors["divisions"].push_back("must be smaller than dimensions");

	// Check the max boundary
	values["maxBoundary"] = std::to_string(instance->maxBoundary);
	if (instance->maxBoundary <= 0)
		errors["maxBoundary"].push_back("must be greater than 0 (adjust dimensions)");

	// Check the number of particles
	values["nParticles"] = std::to_string(instance->nParticles);
	if (instance->nParticles <= 0)
		errors["nParticles"].push_back("must be greater than 0");

	// Check the number of boxes
	values["nBoxes"] = std::to_string(instance->nBoxes);
	if (instance->nBoxes <= 0)
		errors["nBoxes"].push_back("must be greater than 0 (adjust dimensions or divisions)");

	// Check the box size
	values["boxSize"] = std::to_string(instance->boxSize);
	if (instance->boxSize <= 0)
		errors["boxSize"].push_back("must be greater than 0 (adjust dimensions or divisions)");

	// Check the time step
	values["timeStep"] = std::to_string(instance->timeStep);
	if (instance->timeStep <= 0)
		errors["timeStep"].push_back("must be greater than 0");

	// Check the minimum force distance
	values["minForceDistance"] = std::to_string(instance->minForceDistance);
	if (instance->minForceDistance <= 0)
		errors["minForceDistance"].push_back("must be greater than 0");

	// Check the frames per position write
	values["framesPerPosition"] = std::to_string(framesPerPosition);
	if (framesPerPosition < 0)
		errors["framesPerPosition"].push_back("must be positive");

	// Check the frames per state write
	values["framesPerState"] = std::to_string(framesPerState);
	if (framesPerState < 0)
		errors["framesPerState"].push_back("must be positive");

	for (auto& pair : values) {
		if (errors.find(pair.first) == errors.end()) {
			spdlog::info(pair.first + ": " + pair.second);
		}
		else {
			std::string errorString;
			for (auto& error : errors[pair.first]) {
				errorString += "(" + error + ")";
			}
			spdlog::error(pair.first + ": " + pair.second + " " + errorString);
		}
	}

	spdlog::info("---");

	if (!errors.empty())
		throw std::exception("Invalid configuration");
}

Instance* Core::getInstance()
{
	return instance;
}

Particle* Core::getParticles()
{
	return particles;
}

Box* Core::getBoxes()
{
	return boxes;
}

void Core::setInstance(Instance* instance)
{
	if (this->instance)
		delete[] this->instance;

	this->instance = instance;
}

void Core::setParticles(Particle* particles)
{
	if (this->particles)
		delete[] this->particles;

	this->particles = particles;
}

void Core::setBoxes(Box* boxes)
{
	if (this->boxes)
		delete[] this->boxes;

	this->boxes = boxes;
}

void Core::setKernel(std::string kernelName)
{
	this->kernel = Kernel::fromString[kernelName];
	this->kernelName = kernelName;
}

void Core::setFramesPerPosition(int framesPerPosition)
{
	this->framesPerPosition = framesPerPosition;
}

void Core::setFramesPerState(int framesPerState)
{
	this->framesPerState = framesPerState;
}

void Core::writePositionToDisk()
{
	std::ofstream file(PositionDataDirectory / ("position-" + std::to_string(frame) + ".dat"), std::ios::binary);

	for (int i = 0; i < instance->nParticles; i++)
		positionToFile(&file, &particles[i].position.x, &particles[i].position.y);

	file.close();
}

void Core::writeStateToDisk()
{
	std::ofstream file(StateDataDirectory / ("state-" + std::to_string(frame) + ".dat"), std::ios::binary);

	for (int i = 0; i < instance->nParticles; i++)
		particleToFile(&file, &particles[i].position.x, &particles[i].position.y, &particles[i].velocity.x, &particles[i].velocity.y, &particles[i].mass);

	file.close();
}

std::chrono::milliseconds Core::writeToDisk()
{
	auto writeStartTime = getMilliseconds();

	if (frame % framesPerPosition == 0)
		writePositionToDisk();
	if (frame % framesPerState == 0)
		writeStateToDisk();

	return getMilliseconds() - writeStartTime;
}

void Core::run() {
	startTime = getMilliseconds();

	spdlog::info("Initialzing cuda");
	initializeCuda(instance);

	spdlog::info("Running simulation");
	frameTime = getMilliseconds();

	while (alive) {
		// Run the kernel
		auto kernelTime = simulate(instance, particles, boxes, kernel);

		// Write the results to disk
		auto writeTime = writeToDisk();

		spdlog::info("Frame {} completed in {}ms ({}ms kernel, {}ms writing)", frame, (getMilliseconds() - frameTime).count(), kernelTime.count(), writeTime.count());
		frameTime = getMilliseconds();
		frame++;
	}

	spdlog::info("Simulation shutting down");
	unInitializeCuda();
}

void Core::kill() {
	alive = false;
}