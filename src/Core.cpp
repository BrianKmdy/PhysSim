#include <map>
#include <fstream>

#include "spdlog/spdlog.h"

#include "Core.h"
#include "Paths.h"

void dumpParticles(std::string name, int nParticles, Particle* particles)
{
	YAML::Node data;

	data["nParticles"] = nParticles;
	for (int i = 0; i < nParticles; i++) {
		data["particles"][i]["id"] = particles[i].id;
		data["particles"][i]["mass"] = particles[i].mass;
		data["particles"][i]["x"] = particles[i].position.x;
		data["particles"][i]["y"] = particles[i].position.y;
		data["particles"][i]["boxId"] = particles[i].boxId;
	}

	std::ofstream fout(OutputDirectory / name);
	fout << data;
}

void dumpBoxes(std::string name, int nBoxes, Box* boxes)
{
	YAML::Node data;

	data["nBoxes"] = nBoxes;
	for (int i = 0; i < nBoxes; i++) {
		data["boxes"][i]["mass"] = boxes[i].mass;
		data["boxes"][i]["mass x"] = boxes[i].centerMass.x;
		data["boxes"][i]["mass y"] = boxes[i].centerMass.y;
		data["boxes"][i]["nParticles"] = boxes[i].nParticles;
		data["boxes"][i]["particleOffset"] = boxes[i].particleOffset;
	}

	std::ofstream fout(OutputDirectory / name);
	fout << data;
}

void dumpExternalForceField(std::string name, int nExternalForceBoxes, float2* externalForceField)
{
	YAML::Node data;

	for (int i = 0; i < nExternalForceBoxes; i++) {
		data["forcefield"][i]["x"] = std::to_string(externalForceField[i].x);
		data["forcefield"][i]["y"] = std::to_string(externalForceField[i].y);
	}

	std::ofstream fout(OutputDirectory / name);
	fout << data;
}

FrameBufferOut::FrameBufferOut(int queueSize, int nParticles, int framesPerPosition, int framesPerState):
	FrameBuffer<Particle>(queueSize, nParticles, framesPerPosition),
	framesPerPosition(framesPerPosition),
	framesPerState(framesPerState)
{
}

void FrameBufferOut::nextFrame(std::shared_ptr<Particle[]>* frame)
{
	while (framePool.empty()) {
		spdlog::warn("Waiting for a free buffer frame");
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	std::scoped_lock lock(mutex);

	memcpy(framePool.back().get(), frame->get(), nParticles * sizeof(Particle));
	frames[bufferIndex] = framePool.back();
	framePool.pop_back();
	bufferIndex += stepSize;
}

void FrameBufferOut::run()
{
	spdlog::info("Framebuffer thread starting up");

	while (alive) {
		std::shared_ptr<Particle[]> frame = nullptr;

		{
			std::scoped_lock lock(mutex);

			while (!frames.empty() && frameIndex > frames.begin()->first) {
				framePool.push_back(frames.begin()->second);
				frames.erase(frames.begin()->first);
			}

			if (hasFramesBuffered()) {
				frame = frames[frameIndex];
				frameIndex += stepSize;
			}
		}

		// If we have frames available to load and space in the pool then load the next frame
		if (frame) {
			try {
				// Grab the frame and sort by particle id
				std::sort(frame.get(), frame.get() + nParticles,
					[](const Particle& a, const Particle& b) {
					return a.id < b.id;
				});

				// Write the position data to file
				// dumpParticles(("position-" + std::to_string(frameIndex) + ".yaml"), nParticles, frame.get());
				if (frameIndex % framesPerPosition == 0) {
					std::ofstream file(PositionDataDirectory / ("position-" + std::to_string(frameIndex) + ".dat"), std::ios::binary);
					for (int i = 0; i < nParticles; i++)
						positionToFile(&file, &frame[i].position.x, &frame[i].position.y);
					file.close();
				}

				if (frameIndex % framesPerState == 0) {
					std::ofstream file(StateDataDirectory / ("state-" + std::to_string(frameIndex) + ".dat"), std::ios::binary);
					for (int i = 0; i < nParticles; i++)
						particleToFile(&file, &frame[i].position.x, &frame[i].position.y, &frame[i].velocity.x, &frame[i].velocity.y, &frame[i].mass);
					file.close();
				}
			}
			catch (std::exception e) {
				spdlog::error("Unable to read frame {} from file", bufferIndex);
			}
		}
		// Otherwise sleep
		else {
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}

	spdlog::info("Framebuffer thread shutting down");
}

Core::Core():
	Core(nullptr, nullptr, nullptr, nullptr)
{
}

Core::Core(std::shared_ptr<Instance> instance, std::shared_ptr<Particle[]> particles, std::shared_ptr<Box[]> boxes, std::shared_ptr<float2[]> externalForceField):
	alive(true),
	instance(instance),
	particles(particles),
	boxes(boxes),
	externalForceField(externalForceField),
	frame(0),
	framesPerPosition(1),
	framesPerState(100),
	frameBuffer(nullptr),
	kernel(Kernel::unknown),
	kernelName("not set"),
	startTime(std::chrono::milliseconds(0)),
	lastFrameTime(std::chrono::milliseconds(0))
{
}

Core::~Core()
{
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
	if (framesPerState % framesPerPosition != 0)
		errors["framesPerState"].push_back("must be a multiple of framesPerPosition");

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

std::shared_ptr<Instance> Core::getInstance()
{
	return instance;
}

std::shared_ptr<Particle[]> Core::getParticles()
{
	return particles;
}

std::shared_ptr<Box[]> Core::getBoxes()
{
	return boxes;
}

std::shared_ptr<float2[]> Core::getExternalForceField()
{
	return externalForceField;
}

void Core::setInstance(std::shared_ptr<Instance> instance)
{
	this->instance = instance;
}

void Core::setParticles(std::shared_ptr<Particle[]> particles)
{
	this->particles = particles;
}

void Core::setBoxes(std::shared_ptr<Box[]> boxes)
{
	this->boxes = boxes;
}

void Core::setExternalForceField(std::shared_ptr<float2[]> externalForceField)
{
	this->externalForceField = externalForceField;
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

std::chrono::milliseconds Core::writeToDisk()
{
	auto writeStartTime = getMilliseconds();

	if (frame % framesPerPosition == 0 || frame % framesPerState == 0)
		frameBuffer->nextFrame(&particles);

	return getMilliseconds() - writeStartTime;
}

void Core::run()
{
	auto frameTime = std::chrono::milliseconds(0);
	auto kernelTime = std::chrono::milliseconds(0);
	auto writeTime = std::chrono::milliseconds(0);
	int framesThisPeriod = 0;

	frameBuffer = std::make_shared<FrameBufferOut>(MAX_BUFFER_MEMORY / (static_cast<unsigned long long>(instance->nParticles) * sizeof(Particle)), instance->nParticles, framesPerPosition, framesPerState);
	frameBuffer->start();

	startTime = getMilliseconds();

	spdlog::info("Initialzing cuda");
	initializeCuda(instance.get());

	spdlog::info("Running simulation");
	lastFrameTime = getMilliseconds();

	while (alive) {
		// Run the kernel
		kernelTime += simulate(instance.get(), particles.get(), boxes.get(), externalForceField.get(), kernel);
		writeTime += writeToDisk();
		frameTime += (getMilliseconds() - lastFrameTime);

		frame++;
		framesThisPeriod++;

		if (frameTime >= std::chrono::seconds(1) && framesThisPeriod > 0) {
			spdlog::info("[{:.2f}m] Frame {}: {}ms frame ({}ms kernel, {}ms writing)", (getMilliseconds() - startTime).count() / 60000.0f, frame, frameTime.count() / framesThisPeriod, kernelTime.count() / framesThisPeriod, writeTime.count() / framesThisPeriod);
			frameTime = std::chrono::milliseconds(0);
			kernelTime = std::chrono::milliseconds(0);
			writeTime = std::chrono::milliseconds(0);
			framesThisPeriod = 0;
		}

		lastFrameTime = getMilliseconds();
	}

	spdlog::info("Simulation shutting down");
	unInitializeCuda();

	frameBuffer->stop();
	frameBuffer->join();
}

void Core::kill() {
	alive = false;
}