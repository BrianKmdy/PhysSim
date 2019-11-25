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

void dumpState(std::string name)
{
	YAML::Node data;

	Instance* instance = gCore.getInstance();
	Particle* particles = gCore.getParticles();
	Box* boxes = gCore.getBoxes();

	data["nBoxes"] = instance->nBoxes;
	for (int i = 0; i < instance->nBoxes; i++) {
		data["boxes"][i]["mass"] = boxes[i].mass;
		data["boxes"][i]["mass x"] = boxes[i].centerMass.x;
		data["boxes"][i]["mass y"] = boxes[i].centerMass.y;
		data["boxes"][i]["nParticles"] = boxes[i].nParticles;
		data["boxes"][i]["particleOffset"] = boxes[i].particleOffset;
	}

	data["nParticles"] = instance->nParticles;
	for (int i = 0; i < instance->nParticles; i++) {
		data["particles"][i]["mass"] = particles[i].mass;
		data["particles"][i]["x"] = particles[i].position.x;
		data["particles"][i]["y"] = particles[i].position.y;
		data["particles"][i]["boxId"] = particles[i].boxId;
	}

	std::ofstream fout(ConfigFilePath);
	fout << data;
}

Instance* loadInstance()
{
	spdlog::info("Initializing instance");

	// Create the instance for this simulation
	Instance* instance = new Instance;

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

Particle* loadParticles(Instance* instance)
{
	spdlog::info("Initializing particles");

	// Initialize a random number generator for the particles
	std::random_device rd;
	std::mt19937 mt(rd());

	// Create the particles
	Particle* particles = new Particle[instance->nParticles];

	// Initialize the particles based on the config
	int pIndex = 0;
	for (auto it = gConfig["particles"].begin(); it != gConfig["particles"].end(); ++it) {
		YAML::Node node = *it;
		if (node["n"].IsDefined()) {
			float length = node["length"].as<float>();
			std::uniform_real_distribution<float> dist(-length, length);
			for (int i = 0; i < node["n"].as<int>(); i++) {
				particles[pIndex].position = make_float2(node["x"].as<float>() + dist(mt), node["y"].as<float>() + dist(mt));
				particles[pIndex].velocity = make_float2(node["vx"].as<float>(), node["vy"].as<float>());
				particles[pIndex].mass = node["mass"].as<float>();

				pIndex++;
			}
		}
		else {
			particles[pIndex].position = make_float2(node["x"].as<float>(), node["y"].as<float>());
			particles[pIndex].mass = node["mass"].as<float>();
			particles[pIndex].velocity = make_float2(node["vx"].as<float>(), node["vy"].as<float>());

			pIndex++;
		}
	}

	return particles;
}

Box* loadBoxes(Instance* instance)
{
	spdlog::info("Initializing boxes");

	// Initialize the boxes
	Box* boxes = new Box[instance->nBoxes];
	memset(boxes, 0, instance->nBoxes * sizeof(Box));

	return boxes;
}

void loadConfig()
{
	spdlog::info("Loading configuration");
	gConfig = YAML::LoadFile("config.yaml");

	if (!gConfig["name"].IsDefined())
		throw std::exception("name must be defined");

	if (gConfig["kernel"].IsDefined())
		gCore.setKernel(gConfig["kernel"].as<std::string>());
	if (gConfig["framesPerPosition"].IsDefined())
		gCore.setFramesPerPosition(gConfig["framesPerPosition"].as<int>());
	if (gConfig["framesPerState"].IsDefined())
		gCore.setFramesPerState(gConfig["framesPerState"].as<int>());

	// Load the instance
	Instance* instance = loadInstance();
	gCore.setInstance(instance);
	gCore.verifyConfiguration();

	// Load the particles
	Particle* particles = loadParticles(instance);
	gCore.setParticles(particles);

	// Load the boxes
	Box* boxes = loadBoxes(instance);
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
	if (std::filesystem::exists(OutputDirectory)) {
		throw std::exception("Output directory already exists");
	}

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
	}
	catch (std::exception& e) {
		spdlog::error("Error intializing simulation: {}", e.what());

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
