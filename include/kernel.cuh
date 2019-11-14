#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Particle.h"

__global__
void bigloop(unsigned int n, unsigned int deviceBatchSize, int deviceId, unsigned int endIndex, float* data_in, float* data_out);
void test(unsigned int n, unsigned int deviceBatchSize, int deviceId, float* data_in, float* data_out);

__host__ __device__ int test_math();

__host__ __device__ class testclass
{
public:
	std::vector<int> get()
	{
		std::vector<int> test;
		test.push_back(0);
		test.push_back(1);
		test.push_back(2);
		return test;
	}
};