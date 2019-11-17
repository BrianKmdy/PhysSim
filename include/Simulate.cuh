#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

#include "Types.cuh"

struct Particle
{
	float2 position;
	float2 velocity;

	float mass;
};

struct Box
{
	float mass;
	float2 centerMass;

	int nParticles;
	Particle* particles;
};

struct Instance
{
	int dimensions;
	int divisions;

	int nParticles;
	Particle* particles;

	int boxSize;
	int nBoxes;
	Box* boxes;

	__host__ __device__ int getBoxIndex(float2 position);

	__host__ unsigned int size();
	__host__ static unsigned int size(int nParticles, int nBoxes);
};

__global__
void bigloop(unsigned int n, unsigned int deviceBatchSize, int deviceId, unsigned int endIndex, float* data_in, float* data_out);
void test(unsigned int n, unsigned int deviceBatchSize, int deviceId, float* data_in, float* data_out);

__host__ void initialize(Instance* instance);
__host__ void unInitialize();

__host__ void initializeInstance(Instance* instance);

__host__ void simulate(Instance* instance);
__global__ void kernel(int deviceId, unsigned int deviceBatchSize, unsigned int endIndex, Instance* instance);