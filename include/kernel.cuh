#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Particle.h"



class particle
{
	float mass;

	float2 position;
	float2 velocity;
};

class Box
{
	float mass;

	int2 position;

	int numParticles;
	particle* particles;
};


class testclass
{
public:
	testclass():
		test()
	{}

	__host__ std::vector<int> get();
	__host__ __device__ void set();

	std::vector<int> test;
	float3 f;
};

__global__
void bigloop(unsigned int n, unsigned int deviceBatchSize, int deviceId, unsigned int endIndex, float* data_in, float* data_out);
void test(unsigned int n, unsigned int deviceBatchSize, int deviceId, float* data_in, float* data_out);

__global__ void test_math(testclass* test);
void test_math_wrapper(testclass* test);