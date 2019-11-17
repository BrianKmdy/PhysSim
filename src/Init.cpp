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
#include "spdlog/sinks/rotating_file_sink.h"

#include "yaml-cpp/yaml.h"

#include <thread>

const unsigned int size = 10000000;

Core gCore;

void signalHandler(int signum) {
	spdlog::info("Terminate signal received by user");
	gCore.kill();
}

void print_data(float* data)
{
	for (unsigned int i = 0; i < 10; i++)
		std::cout << data[i] << " ";
	std::cout << "... ";
	for (unsigned int i = size - 10; i < size; i++)
		std::cout << data[i] << " ";
	std::cout << std::endl;
	std::cout << std::endl;
}

void test_cuda()
{
	std::cout << "Setting cuda device" << std::endl;

	int nDevices;
	cudaGetDeviceCount(&nDevices);

	unsigned int deviceBatchSize = (size + nDevices - 1) / nDevices;

	std::cout << "Num elements: " << size << std::endl;
	std::cout << "Batch size: " << deviceBatchSize << std::endl;

	float* data_in_h = new float[size];
	float* data_out_h = new float[size];
	memset(data_out_h, 0, size);

	for (unsigned int i = 0; i < size; i++)
	{
		data_in_h[i] = 1.0 / double(i + 1);
	}

	print_data(data_in_h);

	auto time = std::chrono::high_resolution_clock::now();

	float* data_in_d;
	std::vector<float*> data_out_d;
	for (int i = 0; i < nDevices; i++)
	{
		cudaSetDevice(i);
		std::cout << "Allocating memory for input data" << std::endl;
		data_out_d.push_back(0);

		cudaMalloc(&data_in_d, size * sizeof(float));
		cudaMalloc(&data_out_d[i], size * sizeof(float));

		cudaMemcpy(data_in_d, data_in_h, size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(data_out_d[i], data_out_h, size * sizeof(float), cudaMemcpyHostToDevice);

		std::cout << "Starting kernel on device " << std::to_string(i) << std::endl;
		test(size, deviceBatchSize, i, data_in_d, data_out_d[i]);
	}

	for (int i = 0; i < nDevices; i++)
	{
		cudaSetDevice(i);

		std::cout << "Syncing with device " << std::to_string(i) << std::endl;

		cudaDeviceSynchronize();

		unsigned int num = std::min(deviceBatchSize, size - (i * deviceBatchSize));
		cudaMemcpy(data_out_h + (i * deviceBatchSize), data_out_d[i] + (i * deviceBatchSize), num * sizeof(float), cudaMemcpyDeviceToHost);
	}

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - time);
	std::cout << "Total time: " << duration.count() << "ms" << std::endl;
	std::cout << std::endl;
	
	print_data(data_out_h);

	cudaDeviceReset();
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
	Instance* instance = reinterpret_cast<Instance*>(new char[Instance::size(nParticles, nBoxes)]);
	memset(instance, 0, Instance::size(nParticles, nBoxes));

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
	Particle* particles = instance->getParticles();
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
}

void dumpState(Instance* instance, std::string name)
{
	YAML::Node data;

	Box* boxes = instance->getBoxes();
	data["nBoxes"] = instance->nBoxes;
	for (int i = 0; i < instance->nBoxes; i++) {
		data["boxes"][i]["mass"] = boxes[i].mass;
		data["boxes"][i]["mass x"] = boxes[i].centerMass.x;
		data["boxes"][i]["mass y"] = boxes[i].centerMass.y;
		data["boxes"][i]["nParticles"] = boxes[i].nParticles;

		for (int o = 0; o < boxes[i].nParticles; o++) {
			Particle* boxParticles = instance->getBoxParticles();
			data["boxes"][i][o]["mass"] = boxParticles[boxes[i].particleOffset + o].mass;
			data["boxes"][i][o]["x"] = boxParticles[boxes[i].particleOffset + o].position.x;
			data["boxes"][i][o]["y"] = boxParticles[boxes[i].particleOffset + o].position.y;
		}
	}

	Particle* particles = instance->getParticles();
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
	spdlog::init_thread_pool(8192, 1);
	auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt >();
	auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("simlog.txt", 1024 * 1024 * 10, 3);
	std::vector<spdlog::sink_ptr> sinks{ stdout_sink, rotating_sink };
	auto logger = std::make_shared<spdlog::async_logger>("PhysSim", sinks.begin(), sinks.end(), spdlog::thread_pool(), spdlog::async_overflow_policy::block);
	spdlog::set_default_logger(logger);

	try {
		loadConfig();
	}
	catch (std::exception& e) {
		spdlog::error("Error loading config: {}", e.what());

		return -1;
	}

	try {
		dumpState(gCore.getInstance(), "before");
		gCore.run();
		dumpState(gCore.getInstance(), "after");
	}
	catch (std::exception& e) {
		spdlog::error("Error running simulation: {}", e.what());

		return -1;
	}

	return 0;
}
