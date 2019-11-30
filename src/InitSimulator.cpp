#include <ctime>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <random>
#include <fstream>
#include <csignal>
#include <thread>

#include "Paths.h"
#include "Types.h"
#include "Core.h"

#include "Simulate.cuh"
#include "Operations.cuh"

#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

#include "yaml-cpp/yaml.h"

const static int version = 1;

Core gCore;
YAML::Node gConfig;

void signalHandler(int signum) {
	gCore.kill();
}

std::shared_ptr<Instance> loadInstance()
{
	spdlog::info("Initializing instance");

	// Create the instance for this simulation
	std::shared_ptr<Instance> instance = std::make_shared<Instance>();

	// Set the world configuration
	instance->dimensions = gConfig["dimensions"].as<int>();
	instance->divisions = gConfig["divisions"].as<int>();
	instance->nBoxes = instance->divisions * instance->divisions;
	instance->boxSize = instance->dimensions / instance->divisions;
	instance->maxBoundary = instance->dimensions / 2;

	// Set the timestep and minimum force distance
	if (gConfig["timeStep"].IsDefined())
		instance->timeStep = gConfig["timeStep"].as<float>();
	if (gConfig["minForceDistance"].IsDefined())
		instance->minForceDistance = gConfig["minForceDistance"].as<float>();

	// Count the particles
	instance->nParticles = getNParticles(&gConfig);

	return instance;
}

std::shared_ptr<Particle[]> loadParticles(std::shared_ptr<Instance> instance)
{
	spdlog::info("Initializing particles");

	// Initialize a random number generator for the particles
	std::random_device rd;
	std::mt19937 mt(rd());

	// Create the particles
	std::shared_ptr<Particle[]> particles = std::shared_ptr<Particle[]>(new Particle[instance->nParticles]);

	// Initialize the particles based on the config
	int pId = 0;
	for (auto it = gConfig["particles"].begin(); it != gConfig["particles"].end(); ++it) {
		YAML::Node node = *it;
		if (node["n"].IsDefined()) {
			float length = node["length"].as<float>();
			std::uniform_real_distribution<float> dist(-length / 2, length / 2);
			for (int i = 0; i < node["n"].as<int>(); i++) {
				particles[pId].id = pId;
				particles[pId].position = make_float2(node["x"].as<float>() + dist(mt), node["y"].as<float>() + dist(mt));
				particles[pId].velocity = make_float2(node["vx"].as<float>(), node["vy"].as<float>());
				particles[pId].mass = node["mass"].as<float>();

				pId++;
			}
		}
		else {
			particles[pId].id = pId;
			particles[pId].position = make_float2(node["x"].as<float>(), node["y"].as<float>());
			particles[pId].mass = node["mass"].as<float>();
			particles[pId].velocity = make_float2(node["vx"].as<float>(), node["vy"].as<float>());

			pId++;
		}
	}

	return particles;
}

std::shared_ptr<Box[]> loadBoxes(std::shared_ptr<Instance> instance)
{
	spdlog::info("Initializing boxes");

	// Initialize the boxes
	std::shared_ptr<Box[]> boxes = std::shared_ptr<Box[]>(new Box[instance->nBoxes]);
	memset(boxes.get(), 0, instance->nBoxes * sizeof(Box));

	return boxes;
}

void loadConfig()
{
	spdlog::info("Loading configuration");
	gConfig = YAML::LoadFile(ConfigFileName);

	if (!gConfig["name"].IsDefined())
		throw std::exception("name must be defined");

	if (gConfig["kernel"].IsDefined())
		gCore.setKernel(gConfig["kernel"].as<std::string>());
	if (gConfig["framesPerPosition"].IsDefined())
		gCore.setFramesPerPosition(gConfig["framesPerPosition"].as<int>());
	if (gConfig["framesPerState"].IsDefined())
		gCore.setFramesPerState(gConfig["framesPerState"].as<int>());

	// Load the instance
	std::shared_ptr<Instance> instance = loadInstance();
	gCore.setInstance(instance);
	gCore.verifyConfiguration();

	// Load the particles
	std::shared_ptr<Particle[]> particles = loadParticles(instance);
	gCore.setParticles(particles);

	// Load the boxes
	std::shared_ptr<Box[]> boxes = loadBoxes(instance);
	gCore.setBoxes(boxes);
}

void saveConfig()
{
	spdlog::info("Saving configuration");

	gConfig["version"] = version;

	std::ofstream fout(OutputConfigFilePath);
	fout << gConfig;
}

void createDirectories()
{
	if (std::filesystem::exists(OutputDirectory))
		throw std::exception("output directory already exists");

	std::filesystem::create_directory(OutputDirectory);
	std::filesystem::create_directory(PositionDataDirectory);
	std::filesystem::create_directory(StateDataDirectory);
}

int main()
{
	// Create a signal handler in order to stop the simulation
	signal(SIGINT, signalHandler);

	try {
		// Create the output directory structure
		createDirectories();

		// Initiate the logger
		auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt >();
		auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(LogFilePath.string());
		std::vector<spdlog::sink_ptr> sinks{ stdout_sink, file_sink };
		auto logger = std::make_shared<spdlog::logger>("PhysSim", sinks.begin(), sinks.end());
		spdlog::set_default_logger(logger);

		// Load the config
		loadConfig();
		saveConfig();
	}
	catch (std::exception& e) {
		spdlog::error("Error initializing simulation: {}", e.what());

		return -1;
	}

	try {
		// dumpState("before");
		gCore.run();
		// dumpState("after");

		saveConfig();
	}
	catch (std::exception& e) {
		spdlog::error("Error running simulation: {}", e.what());

		return -1;
	}

	return 0;
}
