#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <chrono>

#include "Types.cuh"

const static constexpr int nThreads = 512;

struct Particle
{
	float2 position;
	float2 velocity;
	float2 force;

	float mass;

	int boxId;

	__host__ __device__ float2 direction(float2 otherPosition);
	__host__ __device__ float dist(float2 otherPosition);
	__host__ __device__ void enforceBoundary(float maxBoundary);
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
	float maxBoundary;

	int nParticles;
	int nBoxes;

	__host__ __device__ int getBoxIndex(float2 position);

	__host__ unsigned int size();
	__host__ static unsigned int size(int nParticles, int nBoxes);
};

__host__ void initializeCuda(Instance* instance);
__host__ void unInitializeCuda();

__host__ std::chrono::milliseconds simulate(Instance* instance, Particle *particles, Box* boxes);
__global__ void kernel(int deviceId, int deviceBatchSize, int endIndex, Instance instance, Particle* particles, Box* boxes);

std::chrono::milliseconds getMilliseconds();

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}