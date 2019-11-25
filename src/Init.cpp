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
#include "Core.h"

#include "Simulate.cuh"
#include "Types.cuh"

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
	// Create the instance for this simulation
	Instance* instance = new Instance;

	instance->dimensions = gConfig["dimensions"].as<int>();
	instance->divisions = gConfig["divisions"].as<int>();
	instance->nBoxes = instance->divisions * instance->divisions;
	instance->boxSize = instance->dimensions / instance->divisions;
	instance->maxBoundary = instance->dimensions / 2;

	return instance;
}

Particle* loadParticles(Instance* instance)
{
	// Initialize a random number generator for the particles
	std::random_device rd;
	std::mt19937 mt(rd());

	// Count the particles
	instance->nParticles = 0;
	for (auto it = gConfig.begin(); it != gConfig.end(); ++it) {
		auto node = it->second;

		if (it->first.as<std::string>() == "particle") {
			instance->nParticles++;
		}
		else if (it->first.as<std::string>() == "particles") {
			instance->nParticles += node["n"].as<int>();
		}
	}

	// Create the particles
	Particle* particles = new Particle[instance->nParticles];
	int index = 0;
	for (auto it = gConfig.begin(); it != gConfig.end(); ++it) {
		auto node = it->second;

		if (it->first.as<std::string>() == "particle") {
			particles[index].position = make_float2(node["x"].as<float>(), node["y"].as<float>());
			particles[index].mass = node["mass"].as<float>();
			particles[index].velocity = make_float2(node["vx"].as<float>(), node["vy"].as<float>());

			index++;
		}
		else if (it->first.as<std::string>() == "particles") {
			float length = node["length"].as<float>();
			std::uniform_real_distribution<float> dist(-length, length);
			for (int i = 0; i < node["n"].as<int>(); i++) {
				particles[index].position = make_float2(node["x"].as<float>() + dist(mt), node["y"].as<float>() + dist(mt));
				particles[index].velocity = make_float2(node["vx"].as<float>(), node["vy"].as<float>());
				particles[index].mass = node["mass"].as<float>();

				index++;
			}
		}
	}

	return particles;
}

Box* loadBoxes(Instance* instance)
{
	// Initialize the boxes
	Box* boxes = new Box[instance->nBoxes];
	memset(boxes, 0, instance->nBoxes * sizeof(Box));

	return boxes;
}

void loadConfig()
{
	spdlog::info("Loading config");
	gConfig = YAML::LoadFile("config.yaml");

	if (gConfig["kernel"].IsDefined())
		gCore.setKernel(gConfig["kernel"].as<std::string>());
	if (gConfig["framesPerPosition"].IsDefined())
		gCore.setFramesPerPosition(gConfig["framesPerPosition"].as<int>());
	if (gConfig["framesPerFullState"].IsDefined())
		gCore.setFramesPerState(gConfig["framesPerState"].as<int>());
	if (gConfig["timeStep"].IsDefined())
		gCore.setTimeStep(gConfig["timeStep"].as<float>());
	if (gConfig["minForceDistance"].IsDefined())
		gCore.setMinForceDistance(gConfig["minForceDistance"].as<float>());

	Instance* instance = loadInstance();
	Particle* particles = loadParticles(instance);
	Box* boxes = loadBoxes(instance);

	gCore.setInstance(instance);
	gCore.setParticles(particles);
	gCore.setBoxes(boxes);
}

void saveConfig()
{
	spdlog::info("Saving config");

	gConfig["version"] = version;

	std::ofstream fout(OutputConfigFilePath);
	fout << gConfig;
}

void createDirectories()
{
	spdlog::info("Creating output directory");

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
