#include <ctime>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <random>
#include <fstream>
#include <csignal>

#include "Core.h"

#include "Simulate.cuh"
#include "Types.cuh"

#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

#include "yaml-cpp/yaml.h"

#include <thread>

Core gCore;

void signalHandler(int signum) {
	spdlog::info("Terminate signal received by user");
	gCore.kill();
}

void loadConfig()
{
	spdlog::info("Loading config");
	YAML::Node config = YAML::LoadFile("config.yaml");

	// Load the environment setup from the config
	int dimensions = config["dimensions"].as<int>();
	int divisions = config["divisions"].as<int>();
	int nParticles = config["nParticles"].as<int>();
	int nBoxes = divisions * divisions;
	int boxSize = dimensions / divisions;

	// Create the instance for this simulation
	Instance* instance = new Instance;
	Particle* particles = new Particle[nParticles];
	Box* boxes = new Box[nBoxes];
	memset(boxes, 0, nBoxes * sizeof(Box));

	instance->dimensions = dimensions;
	instance->divisions = divisions;
	instance->nParticles = nParticles;
	instance->nBoxes = nBoxes;
	instance->boxSize = boxSize;

	float maxBound = dimensions / 2;
	instance->left = -maxBound;
	instance->right = maxBound;
	instance->bottom = -maxBound;
	instance->top = maxBound;

	// Initialize a random number generator for the particles
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> dist(-maxBound, maxBound);

	// Create the particles
	for (int i = 0; i < instance->nParticles; i++) {
		particles[i].position = make_float2(dist(mt), dist(mt));
		particles[i].velocity = make_float2(0.0f, 0.0f);
		particles[i].mass = 1.0f;
	}

	if (config["kernel"].IsDefined()) {
		gCore.setKernel(config["kernel"].as<std::string>());
	}

	if (config["framesPerWrite"].IsDefined())
		gCore.setFramesPerWrite(config["framesPerWrite"].as<int>());

	gCore.setInstance(instance);
	gCore.setParticles(particles);
	gCore.setBoxes(boxes);
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
	}

	data["nParticles"] = instance->nParticles;
	for (int i = 0; i < instance->nParticles; i++) {
		data["particles"][i]["mass"] = particles[i].mass;
		data["particles"][i]["x"] = particles[i].position.x;
		data["particles"][i]["y"] = particles[i].position.y;
	}

	std::ofstream fout(name + ".yaml");
	fout << data;
}

int main()
{
	// Create a signal handler in order to stop the simulation
	signal(SIGINT, signalHandler);

	// XXX/bmoody Review this
	// Initiate the logger
	auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt >();
	auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("simlog.txt");
	std::vector<spdlog::sink_ptr> sinks{ stdout_sink, file_sink };
	auto logger = std::make_shared<spdlog::logger>("PhysSim", sinks.begin(), sinks.end());
	spdlog::set_default_logger(logger);

	try {
		loadConfig();
	}
	catch (std::exception& e) {
		spdlog::error("Error loading config: {}", e.what());

		return -1;
	}

	try {
		dumpState("before");
		gCore.run();
		dumpState("after");
	}
	catch (std::exception& e) {
		spdlog::error("Error running simulation: {}", e.what());

		return -1;
	}

	return 0;
}
