#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <chrono>

#include "Operations.cuh"

const static constexpr int nThreads = 512;

struct Particle
{
	int id;

	float2 position;
	float2 velocity;
	float2 force;

	float mass;

	int boxId;

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

	float timeStep;
	float minForceDistance;

	int nParticles;
	int nBoxes;

	Instance():
		dimensions(0),
		divisions(0),
		boxSize(0),
		maxBoundary(0.0),
		timeStep(1.0),
		minForceDistance(1.0),
		nParticles(0),
		nBoxes(0)
	{
	}

	__host__ __device__ int getBoxIndex(float2 position);
	__host__ __device__ bool adjacentBoxes(int boxId1, int boxId2);
};

// XXX/bmoody Can move all of this into a class
__host__ void initializeCuda(Instance* instance);
__host__ void unInitializeCuda();

__host__ std::chrono::milliseconds simulate(Instance* instance, Particle *particles, Box* boxes, int kernel);
__global__ void gravity(int deviceId, int deviceBatchSize, int endIndex, Instance instance, Particle* particles, Box* boxes);
__global__ void experimental(int deviceId, int deviceBatchSize, int endIndex, Instance instance, Particle* particles, Box* boxes);