#include <algorithm>

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

__host__ std::chrono::milliseconds simulate(Instance* instance, Particle* particles, Box* boxes)
{
	for (int i = 0; i < instance->nBoxes; i++) {
		boxes[i].mass = 0.0;
		boxes[i].centerMass = make_float2(0.0, 0.0);
		boxes[i].nParticles = 0;
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
	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		gpuErrchk(cudaMemcpy(gDeviceParticles[i], particles, instance->nParticles * sizeof(Particle), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(gDeviceBoxes[i], boxes, instance->nBoxes * sizeof(Box), cudaMemcpyHostToDevice));
		kernel<<<(instance->nParticles + nThreads - 1) / nThreads, nThreads>>>(i, 0, 0, *instance, gDeviceParticles[i], gDeviceBoxes[i]);
		gpuErrchk(cudaPeekAtLastError());
	}

	// Synchronize with devices and copy the udpated instance back
	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(particles, gDeviceParticles[i], instance->nParticles * sizeof(Particle), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(boxes, gDeviceBoxes[i], instance->nBoxes * sizeof(Box), cudaMemcpyDeviceToHost));
	}
	auto kernelEndTime = getMilliseconds();

	// XXX/bmoody Can consider moving this outside of the kernel
	// XXX/bmoody Can store force in the particle struct and avoid storing particles 2x (need to test which is faster)
	const float time = 1.0;
	for (int i = 0; i < instance->nParticles; i++) {
		float2 acceleration = particles[i].force / particles[i].mass;

		particles[i].position += (particles[i].velocity * time) + (0.5 * acceleration * powf(time, 2.0));
		particles[i].velocity += acceleration * time;

		//// XXX/bmoody Review this, there must be a better way
		if (particles[i].position.x < instance->left) {
			particles[i].position.x = instance->left;
			particles[i].velocity.x = 0.0;
		}
		if (particles[i].position.x > instance->right) {
			particles[i].position.x = instance->right - 1;
			particles[i].velocity.x = 0.0;
		}
		if (particles[i].position.y < instance->bottom) {
			particles[i].position.y = instance->bottom;
			particles[i].velocity.y = 0.0;
		}
		if (particles[i].position.y > instance->top) {
			particles[i].position.y = instance->top - 1;
			particles[i].velocity.y = 0.0;
		}
	}

	return kernelEndTime - kernelStartTime;
}

__global__ void kernel(int deviceId, unsigned int deviceBatchSize, unsigned int endIndex, Instance instance, Particle* particles, Box* boxes)
{
	const float minForceDistance = 1.0;

	// unsigned int index = deviceId * deviceBatchSize + blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int stride = blockDim.x * gridDim.x;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	// Particle* particles = instance->getParticles();
	// Particle* boxParticles = instance->getBoxParticles();

	for (int i = index; i < instance.nParticles; i += stride) {
	 	for (int o = 0; o < instance.nParticles; o++) {
	 		// XXX/bmoody Can review making this more efficient, is it necessary to square/sqrt dist so much?
	 		float dist = particles[i].dist(&particles[o]);
	 	
	 		if (dist > minForceDistance) {
	 			particles[i].force += (particles[i].direction(&particles[o]) / dist) * ((particles[i].mass * particles[o].mass) / powf(dist, 2.0));
	 		}
	 	}
	}
}

__host__ __device__ int Instance::getBoxIndex(float2 position)
{
	int2 index = (position + (dimensions / 2)) / boxSize;

	return index.x * divisions + index.y;
}

__host__ unsigned int Instance::size()
{
	return size(nParticles, nBoxes);
}

__host__ unsigned int Instance::size(int nParticles, int nBoxes)
{
	return sizeof(Instance) + (2 * sizeof(Particle) * nParticles) + (sizeof(Box) * nBoxes);
}

__host__ __device__ float2 Particle::direction(Particle* particle)
{
	return particle->position - position;
}

// XXX/bmoody Review the order
//            Can probably make this more efficient by skippint the sqrt
__host__ __device__ float Particle::dist(Particle* particle)
{
	return sqrtf(powf(position.x - particle->position.x, 2.0) + powf(position.y - particle->position.y, 2.0));
}

std::chrono::milliseconds getMilliseconds()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
}