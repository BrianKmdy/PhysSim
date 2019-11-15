#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

class Particle
{
	float2 position;
	float2 velocity;

	float mass;
};

class Box
{

	int2 position;

	float mass;
	float2 centerMass;

	int numParticles;
	Particle* particles;
};


__global__
void bigloop(unsigned int n, unsigned int deviceBatchSize, int deviceId, unsigned int endIndex, float* data_in, float* data_out);
void test(unsigned int n, unsigned int deviceBatchSize, int deviceId, float* data_in, float* data_out);
