#include <algorithm>

#include "Types.h"
#include "Simulate.cuh"

std::vector<Particle*> gDeviceParticles;
std::vector<Box*> gDeviceBoxes;

__host__ void initializeCuda(Instance* instance)
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		gDeviceParticles.push_back(nullptr);
		cudaMalloc(&gDeviceParticles[i], instance->nParticles * sizeof(Particle));
		gDeviceBoxes.push_back(nullptr);
		cudaMalloc(&gDeviceBoxes[i], instance->nBoxes * sizeof(Box));
	}
}

__host__ void unInitializeCuda()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		cudaFree(gDeviceParticles[i]);
		cudaFree(gDeviceBoxes[i]);
		cudaDeviceReset();
	}
}

__host__ std::chrono::milliseconds simulate(Instance* instance, Particle* particles, Box* boxes, int kernel)
{
	for (int i = 0; i < instance->nBoxes; i++) {
		boxes[i].mass = 0.0;
		boxes[i].centerMass = make_float2(0.0, 0.0);
		boxes[i].nParticles = 0;
		boxes[i].particleOffset = 0;
	}

	for (int i = 0; i < instance->nParticles; i++) {
		particles[i].force = make_float2(0.0, 0.0);
		particles[i].boxId = instance->getBoxIndex(particles[i].position);
		boxes[particles[i].boxId].centerMass = (boxes[particles[i].boxId].centerMass * boxes[particles[i].boxId].mass + particles[i].position * particles[i].mass)
			/ (boxes[particles[i].boxId].mass + particles[i].mass);
		boxes[particles[i].boxId].mass += particles[i].mass;
		boxes[particles[i].boxId].nParticles += 1;
	}

	// Is it possible to do this in a more efficient way without using a sort?
	std::sort(particles, particles + instance->nParticles,
		[](const Particle& a, const Particle& b) {
			return a.boxId < b.boxId;
	});

	int boxId = -1;
	for (int i = 0; i < instance->nParticles; i++) {
		if (particles[i].boxId != boxId) {
			boxes[particles[i].boxId].particleOffset = i;
			boxId = particles[i].boxId;
		}
	}

	int nDevices;
	cudaGetDeviceCount(&nDevices);

	// Copy the instance to device memory and run the kernel
	auto kernelStartTime = getMilliseconds();
	int blockSize = (instance->nParticles + nThreads - 1) / nThreads;
	int deviceBatchSize = (instance->nParticles + nDevices - 1) / nDevices;
	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		int endIndex = static_cast<int>(std::min((i + 1) * deviceBatchSize, instance->nParticles));
		gpuErrchk(cudaMemcpy(gDeviceParticles[i], particles, instance->nParticles * sizeof(Particle), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(gDeviceBoxes[i], boxes, instance->nBoxes * sizeof(Box), cudaMemcpyHostToDevice));

		// Launch the kernel
		switch (kernel) {
			case Kernel::experimental:
				experimental<<<blockSize, nThreads>>>(i, deviceBatchSize, endIndex, *instance, gDeviceParticles[i], gDeviceBoxes[i]);
				break;
			default:
				gravity<<<blockSize, nThreads>>>(i, deviceBatchSize, endIndex, *instance, gDeviceParticles[i], gDeviceBoxes[i]);
				break;
		}

		gpuErrchk(cudaPeekAtLastError());
	}

	// Synchronize with devices and copy the udpated instance back
	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		int numElements = std::min(deviceBatchSize, instance->nParticles - (i * deviceBatchSize));
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(particles + (i * deviceBatchSize), gDeviceParticles[i] + (i* deviceBatchSize), numElements * sizeof(Particle), cudaMemcpyDeviceToHost));
	}
	auto kernelEndTime = getMilliseconds();

	// Calculate each particle's new position and velocity based on its force for this frame
	for (int i = 0; i < instance->nParticles; i++) {
		float2 acceleration = particles[i].force / particles[i].mass;

		particles[i].position += (particles[i].velocity * instance->timeStep) + (0.5 * acceleration * powf(instance->timeStep, 2.0));
		particles[i].velocity += acceleration * instance->timeStep;
		particles[i].enforceBoundary(instance->maxBoundary);
	}

	return kernelEndTime - kernelStartTime;
}

__global__ void gravity(int deviceId, int deviceBatchSize, int endIndex, Instance instance, Particle* particles, Box* boxes)
{
	unsigned int index = deviceId * deviceBatchSize + blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (int i = index; i < endIndex; i += stride) {
	 	for (int o = 0; o < instance.nBoxes; o++) {
			if (o == particles[i].boxId) {
				for (int p = boxes[o].particleOffset; p < boxes[o].particleOffset + boxes[o].nParticles; p++) {
					// XXX/bmoody Can review making this more efficient, is it necessary to square/sqrt dist so much?
					float dist = distance(particles[i].position, particles[p].position);
					if (dist > instance.minForceDistance)
						particles[i].force += (direction(particles[i].position, particles[p].position) / dist) * ((particles[i].mass * particles[p].mass) / powf(dist, 2.0));
				}
			}
			else
			{
				float dist = distance(particles[i].position, boxes[o].centerMass);
				if (dist > instance.minForceDistance)
					particles[i].force += (direction(particles[i].position, boxes[o].centerMass) / dist) * ((particles[i].mass * boxes[o].mass) / powf(dist, 2.0));
			}
	 	}
	}
}

// XXX/bmoody Can make this more accurate (expensive) by having each particle directly interact with the particles of adjacent boxes as well
__global__ void experimental(int deviceId, int deviceBatchSize, int endIndex, Instance instance, Particle* particles, Box* boxes)
{
	unsigned int index = deviceId * deviceBatchSize + blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (int i = index; i < endIndex; i += stride) {
		for (int o = 0; o < instance.nBoxes; o++) {
			if (o == particles[i].boxId) {
				for (int p = boxes[o].particleOffset; p < boxes[o].particleOffset + boxes[o].nParticles; p++) {
					// XXX/bmoody Can review making this more efficient, is it necessary to square/sqrt dist so much?
					float dist = distance(particles[i].position, particles[p].position);
					if (dist > instance.minForceDistance)
						particles[i].force += (direction(particles[p].position, particles[i].position) / dist) * ((particles[i].mass * particles[p].mass) / powf(dist, 2.0));
				}
			}
			else
			{
				float dist = distance(particles[i].position, boxes[o].centerMass);
				if (dist > instance.minForceDistance)
					particles[i].force += (direction(boxes[o].centerMass, particles[i].position) / dist) * ((particles[i].mass * boxes[o].mass) / powf(dist, 2.0));
			}
		}
	}
}

__host__ __device__ int Instance::getBoxIndex(float2 position)
{
	int2 index = (position + (dimensions / 2)) / boxSize;

	return index.x * divisions + index.y;
}

__host__ __device__ void Particle::enforceBoundary(float maxBoundary)
{
	//// XXX/bmoody Review this, there must be a better way
	if (position.x < -maxBoundary) {
		position.x = -maxBoundary + 1.0;
		velocity.x = 0.0;
	}
	if (position.x > maxBoundary) {
		position.x = maxBoundary - 1.0;
		velocity.x = 0.0;
	}
	if (position.y < -maxBoundary) {
		position.y = -maxBoundary + 1.0;
		velocity.y = 0.0;
	}
	if (position.y > maxBoundary) {
		position.y = maxBoundary - 1.0;
		velocity.y = 0.0;
	}
}