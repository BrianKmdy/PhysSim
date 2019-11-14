#include <ctime>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <string>

#include "Particle.h"
#include "Core.h"

#include "kernel.cuh"
#include "types.cuh"

#include "yaml-cpp/yaml.h"

#include <thread>

const unsigned int size = 10000000;

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

int main()
{
	// float* test = new float[size];
	// test_loop(test);

	YAML::Node config = YAML::LoadFile("config.yaml");
	for (YAML::const_iterator it = config.begin(); it != config.end(); ++it) {
		std::cout << it->first.as<std::string>() << ": " << it->second.as<std::string>() << "\n";
	}

	auto test = testclass();
	test_math_wrapper(&test);
	for (auto num : test.get())
		std::cout << num << std::endl;

	float3 t = { 1.0, 2.0, 3.0 };
	float3 z = { 4.0, 5.0, 6.0 };
	
	printf3(t);
	printf3(z);

	printf3(t * z);
	printf3(z - t);



	std::cin.get();

	return 0;
}
