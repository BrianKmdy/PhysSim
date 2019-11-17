#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

#include "Types.cuh"

const static constexpr int nThreads = 512;

struct Particle
{
	float2 position;
	float2 velocity;

	float mass;

	__host__ __device__ float2 direction(Particle* particle);
	__host__ __device__ float dist(Particle* particle);
};

struct Box
{
	float mass;
	float2 centerMass;

	int nParticles;
	int particleOffset;
};

struct Instance
{
	int dimensions;
	int divisions;
	int boxSize;

	int nParticles;
	int nBoxes;

	float left;
	float right;
	float bottom;
	float top;

	__host__ __device__ int getBoxIndex(float2 position);

	__host__ unsigned int size();
	__host__ static unsigned int size(int nParticles, int nBoxes);

	__host__ __device__ Particle* getParticles();
	__host__ __device__ Box* getBoxes();
	__host__ __device__ Particle* getBoxParticles(int particleOffset = 0);
};

__global__
void bigloop(unsigned int n, unsigned int deviceBatchSize, int deviceId, unsigned int endIndex, float* data_in, float* data_out);
void test(unsigned int n, unsigned int deviceBatchSize, int deviceId, float* data_in, float* data_out);

__host__ void initializeCuda(Instance* instance);
__host__ void unInitializeCuda();

__host__ void simulate(Instance* instance);
__global__ void kernel(int deviceId, unsigned int deviceBatchSize, unsigned int endIndex, Instance* instance);