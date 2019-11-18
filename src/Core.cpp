#include <map>
#include <fstream>

#include "spdlog/spdlog.h"

#include "Core.h"

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
	framesPerWrite(1),
	kernel(Kernel::unknown),
	kernelName("not set"),
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

void Core::setFramesPerWrite(int framesPerWrite)
{
	this->framesPerWrite = framesPerWrite;
}

void Core::verifyConfiguration()
{
	if (!instance)
		throw std::exception("No instance set");

	spdlog::info("---");
	spdlog::info("Config settings");

	std::map<std::string, std::string> values;
	std::map<std::string, std::vector<std::string>> errors;

	// Check the dimensions
	values["dimensions"] = std::to_string(instance->dimensions);
	if (instance->dimensions & (instance->dimensions - 1))
		errors["dimensions"].push_back("must be a power of 2");
	
	// Check the divisions
	values["divisions"] = std::to_string(instance->divisions);
	if (instance->divisions & (instance->divisions - 1))
		errors["divisions"].push_back("must be a power of 2");

	values["nParticles"] = std::to_string(instance->nParticles);
	values["nBoxes"] = std::to_string(instance->nBoxes);

	// Check the kernel
	values["kernel"] = kernelName;
	if (kernel == Kernel::unknown)
		errors["kernel"].push_back("invalid kernel");

	// Check the frames per write
	values["framesPerWrite"] = std::to_string(framesPerWrite);
	if (framesPerWrite < 0)
		errors["framesPerWrite"].push_back("must be positive");

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

std::chrono::milliseconds Core::writeToDisk()
{
	// If it's not time to write then just return 0ms
	if (frame % framesPerWrite != 0)
		return std::chrono::milliseconds(0);

	auto startTime = getMilliseconds();

	std::ofstream file("test_frame_" + std::to_string(frame) + ".dat", std::ios::binary);

	file.write(reinterpret_cast<char*>(&instance->nParticles), sizeof(int));

	for (int i = 0; i < instance->nParticles; i++)
		file.write(reinterpret_cast<char*>(&particles[i].position), sizeof(float2));
	
	file.close();

	return getMilliseconds() - startTime;
}

void Core::run() {
	verifyConfiguration();

	spdlog::info("Initialzing cuda");
	initializeCuda(instance);

	spdlog::info("Running simulation");
	frameTime = getMilliseconds();

	while (alive) {
		// Run the kernel
		auto kernelTime = simulate(instance, particles, boxes);

		// Write the results to disk
		auto writeTime = writeToDisk();

		spdlog::info("Frame {} completed in {}ms ({}ms kernel, {}ms writing)", frame, (getMilliseconds() - frameTime).count(), kernelTime.count(), writeTime.count());
		frameTime = getMilliseconds();
		frame++;
	}

	spdlog::info("Shutting down");
	unInitializeCuda();
}

void Core::kill() {
	alive = false;
}