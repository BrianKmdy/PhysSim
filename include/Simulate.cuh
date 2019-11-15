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
	int2 position;

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

	int nBoxParticles;
	Particle* boxParticles;

	int nBoxes;
	Box* boxes;
};

__global__
void bigloop(unsigned int n, unsigned int deviceBatchSize, int deviceId, unsigned int endIndex, float* data_in, float* data_out);
void test(unsigned int n, unsigned int deviceBatchSize, int deviceId, float* data_in, float* data_out);

__host__ void initialize(Instance* instance);
__host__ void unInitialize();

__host__ void simulate(Instance* instance);
__global__ void kernel();
__device__ int getBox();