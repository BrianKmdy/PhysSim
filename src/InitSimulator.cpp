#include <ctime>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <random>
#include <fstream>
#include <csignal>
#include <thread>

#include "cxxopts.hpp"

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
bool gResume = false;

void signalHandler(int signum) {
	gCore.kill();
}

std::shared_ptr<Instance> createInstance()
{
	spdlog::info("Initializing instance");

	// Create the instance for this simulation
	std::shared_ptr<Instance> instance = std::make_shared<Instance>();

	// Set the world configuration
	instance->dimensions = gConfig["dimensions"].as<uint32_t>();
	instance->divisions = gConfig["divisions"].as<uint32_t>();
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
	spdlog::info("Loading saved particle state from disk");

	if (!std::filesystem::exists(StateDataDirectory))
		throw std::runtime_error("Unable to find saved state directory");

	// Create the particles
	std::shared_ptr<Particle[]> particles = std::shared_ptr<Particle[]>(new Particle[instance->nParticles]);

	int latestFrame = 0;
	std::string latestState;
	for (auto& path : std::filesystem::directory_iterator(StateDataDirectory)) {
		std::string stringPath = path.path().string();
		int frame = std::stoi(stringPath.substr(stringPath.find('-') + 1, stringPath.find('.')));
		if (frame > latestFrame) {
			latestFrame = frame;
			latestState = stringPath;
		}
	}

	if (latestFrame == 0)
		throw std::runtime_error("No state files found");

	spdlog::info("Latest state file found: {}", latestState);

	// spdlog::info("Loading file {}", files[bufferIndex]);
	int id = 0;
	std::ifstream stateFile(latestState, std::ios_base::in | std::ios_base::binary);
	for (int i = 0; i < instance->nParticles; i++) {
		particleFromFile(&stateFile, &particles[i].position.x, &particles[i].position.y, &particles[i].velocity.x, &particles[i].velocity.y, &particles[i].mass);
		particles[i].id = id++;
	}
	stateFile.close();

	spdlog::info("Setting core to frame {}", latestFrame);
	gCore.setFrame(latestFrame);

	return particles;
}

std::shared_ptr<Particle[]> createParticles(std::shared_ptr<Instance> instance)
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
			for (int i = 0; i < node["n"].as<uint32_t>(); i++) {
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

std::shared_ptr<Box[]> createBoxes(std::shared_ptr<Instance> instance)
{
	spdlog::info("Initializing boxes");

	// Initialize the boxes
	std::shared_ptr<Box[]> boxes = std::shared_ptr<Box[]>(new Box[instance->nBoxes]);
	memset(boxes.get(), 0, instance->nBoxes * sizeof(Box));

	return boxes;
}

std::shared_ptr<float2[]> createForceField(std::shared_ptr<Instance> instance, std::shared_ptr<Particle[]> particles)
{
	std::shared_ptr<float2[]> externalForceField;

	float totalMass = 0.0f;
	for (int i = 0; i < instance->nParticles; i++)
		totalMass += particles[i].mass;

	if (gConfig["externalForceFieldDivisions"].IsDefined()) {
		spdlog::info("Generating external force field");

		instance->externalForceFieldDivisions = gConfig["externalForceFieldDivisions"].as<uint32_t>();
		instance->externalForceBoxSize = instance->dimensions / instance->externalForceFieldDivisions;
		instance->nExternalForceBoxes = instance->externalForceFieldDivisions * instance->externalForceFieldDivisions;

		externalForceField = std::shared_ptr<float2[]>(new float2[instance->nExternalForceBoxes]);
		memset(externalForceField.get(), 0, instance->nExternalForceBoxes * sizeof(float2));

		float averageMass = totalMass / instance->nExternalForceBoxes;
		float averageDist = 0.0f;
		auto startTime = getMilliseconds();

		for (int i = 0; i < instance->nExternalForceBoxes; i++) {
			averageDist = 0.0f;
			float2 totalForce = make_float2(0.0f, 0.0f);
			for (int o = 0; o < instance->nExternalForceBoxes; o++) {
				if (i == o)
					continue;

				float dist = distance(instance->externalForceBoxCenter(i), instance->externalForceBoxCenter(o));
				float2 directionUnit = (direction(instance->externalForceBoxCenter(i), instance->externalForceBoxCenter(o)) / dist);
				float mass = (averageMass / powf(dist, 2.0));
				float2 force = directionUnit * mass;
				totalForce += force;
				averageDist = (averageDist * o + dist) / (o + 1);

				if ((getMilliseconds() - startTime) > std::chrono::seconds(1)) {
					spdlog::info("Generating field: {:.0f}%", static_cast<float>(i) / static_cast<float>(instance->nExternalForceBoxes) * 100.0f);
					startTime = getMilliseconds();
				}
			}

			externalForceField[i] = totalForce;
		}
	}
	else {
		externalForceField = std::shared_ptr<float2[]>(new float2[1]);
	}

	return externalForceField;
}

void loadConfig()
{
	std::string configPath = ConfigFileName;
	if (gResume) {
		spdlog::info("Resuming latest simulation");
		configPath = OutputConfigFilePath.string();
	}

	spdlog::info("Loading configuration");
	gConfig = YAML::LoadFile(configPath);

	if (!gConfig["name"].IsDefined())
		throw std::runtime_error("name must be defined");

	if (gConfig["kernel"].IsDefined())
		gCore.setKernel(gConfig["kernel"].as<std::string>());
	if (gConfig["framesPerPosition"].IsDefined())
		gCore.setFramesPerPosition(gConfig["framesPerPosition"].as<uint32_t>());
	if (gConfig["framesPerState"].IsDefined())
		gCore.setFramesPerState(gConfig["framesPerState"].as<uint32_t>());

	if (gConfig["maxUtilization"].IsDefined())
		gCore.setMaxUtilization(gConfig["maxUtilization"].as<float>());

	// Create the instance
	std::shared_ptr<Instance> instance = createInstance();
	gCore.setInstance(instance);
	gCore.verifyConfiguration();

	// Create the particles
	std::shared_ptr<Particle[]> particles;
	if (gResume)
		particles = loadParticles(instance);
	else
		particles = createParticles(instance);
	gCore.setParticles(particles);

	// Create the boxes
	std::shared_ptr<Box[]> boxes = createBoxes(instance);
	gCore.setBoxes(boxes);

	// Create the external force field
	std::shared_ptr<float2[]> externalForceField = createForceField(instance, particles);
	gCore.setExternalForceField(externalForceField);
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
		throw std::runtime_error("output directory already exists");

	std::filesystem::create_directory(OutputDirectory);
	std::filesystem::create_directory(PositionDataDirectory);
	std::filesystem::create_directory(StateDataDirectory);
}

void createLogger()
{
	auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt >();
	auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(LogFilePath.string());
	std::vector<spdlog::sink_ptr> sinks{ stdout_sink, file_sink };
	auto logger = std::make_shared<spdlog::logger>("PhysSim", sinks.begin(), sinks.end());
	spdlog::set_default_logger(logger);
}

int main(int argc, char* argv[])
{
	// Create a signal handler in order to stop the simulation
	signal(SIGINT, signalHandler);

	try {
		cxxopts::Options options(argv[0], "Post processing and visualization of particle physics simulations");
		options
			.add_options()
			("help", "Print help")
			("r, resume", "Resume the latest simulation", cxxopts::value<bool>());

		auto result = options.parse(argc, argv);

		if (result.count("help")) {
			std::cout << options.help() << std::endl;
			exit(0);
		}

		if (result.count("resume"))
			gResume = true;

		// Create the output directory structure
		if (!gResume)
			createDirectories();

		// Once the output folder exists we can initialize the logger
		createLogger();

		// Load the state from the latest run
		loadConfig();
		saveConfig();
	}
	catch (const cxxopts::OptionException& e) {
		spdlog::error("Error parsing arguments: {}", e.what());
		return -1;
	}
	catch (std::exception& e) {
		spdlog::error("Error initializing simulation: {}", e.what());
		return -1;
	}

	try {
		gCore.run();

		saveConfig();
	}
	catch (std::exception& e) {
		spdlog::error("Error running simulation: {}", e.what());
		return -1;
	}

	return 0;
}
