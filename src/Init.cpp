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
#include "yaml-cpp/yaml.h"

#include <thread>

const unsigned int size = 10000000;

Core gCore;

void signalHandler(int signum) {
	gCore.kill();
}

void load_yaml()
{

}

void bigloop_cpu(unsigned int n, float* data_in, float* data_out)
{
	for (unsigned int i = 0; i < n; i += 1) {
		float lol = 0.0;
		for (unsigned int o = 0; o < n; o++) {
			lol += data_in[o];
		}

		data_out[i] = lol;
	}
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

void test_cpu()
{
	std::cout << "Allocating memory for input data" << std::endl;
	float* data_in = new float[size];
	std::cout << "Allocating memory for output data" << std::endl;
	float* data_out = new float[size];

	for (unsigned int i = 0; i < size; i++)
	{
		data_in[i] = 1.0 / float(i + 1);
	}

	print_data(data_in);

	std::cout << "Running test" << std::endl;
	auto time = std::chrono::high_resolution_clock::now();
	bigloop_cpu(size, data_in, data_out);

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - time);
	std::cout << "Total time: " << duration.count() << "ms" << std::endl;
	std::cout << std::endl;

	print_data(data_out);
}

void test_loop(float* test)
{
	std::cout << "Running test" << std::endl;
	auto time = std::chrono::high_resolution_clock::now();

	float lol = 0.0;
	for (int i = 0; i < size; i++)
	{
		lol += 1.0;
		test[i] = lol;
	}

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - time);
	std::cout << "Total time: " << duration.count() << "ms" << std::endl;
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

void printf3(float3 f3)
{
	std::cout << "x: " << f3.x << " ";
	std::cout << "y: " << f3.y << " ";
	std::cout << "z: " << f3.z << " ";
	std::cout << std::endl;
}

Instance* loadConfig()
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

	initializeInstance(instance);

	// Initialize a random number generator for the particles
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> dist((dimensions / 2) * -1, dimensions / 2);

	// Create the particles
	for (int i = 0; i < instance->nParticles; i++) {
		instance->particles[i].position = make_float2(dist(mt), dist(mt));
		instance->particles[i].velocity = make_float2(0.0f, 0.0f);
		instance->particles[i].mass = 1.0f;
	}

	return instance;
}

void dumpState(Instance* instance, std::string name)
{
	YAML::Node data;

	data["nBoxes"] = instance->nBoxes;
	for (int i = 0; i < instance->nBoxes; i++) {
		data["boxes"][i]["mass"] = instance->boxes[i].mass;
		data["boxes"][i]["mass x"] = instance->boxes[i].centerMass.x;
		data["boxes"][i]["mass y"] = instance->boxes[i].centerMass.y;
		data["boxes"][i]["nParticles"] = instance->boxes[i].nParticles;

		for (int o = 0; o < instance->boxes[i].nParticles; o++) {
			data["boxes"][i][o]["mass"] = instance->boxes[i].particles[o].mass;
			data["boxes"][i][o]["x"] = instance->boxes[i].particles[o].position.x;
			data["boxes"][i][o]["y"] = instance->boxes[i].particles[o].position.y;
		}
	}

	data["nParticles"] = instance->nParticles;
	for (int i = 0; i < instance->nParticles; i++) {
		data["particles"][i]["mass"] = instance->particles[i].mass;
		data["particles"][i]["x"] = instance->particles[i].position.x;
		data["particles"][i]["y"] = instance->particles[i].position.y;
	}

	std::ofstream fout(name + ".yaml");
	fout << data;
}

int main()
{
	// Create the signal handler in order to stop the simulation
	signal(SIGINT, signalHandler);

	try {
		gCore.setInstance(loadConfig());
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
