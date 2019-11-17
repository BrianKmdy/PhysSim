#include "Simulate.cuh"

std::vector<Instance*> gDeviceInstances;

__global__
void bigloop(unsigned int n, unsigned int deviceBatchSize, int deviceId, unsigned int endIndex, float* data_in, float* data_out)
{
	unsigned int index = deviceId * deviceBatchSize + blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = index; i < endIndex; i += stride) {
		float lol = 0.0;
		for (unsigned int o = 0; o < n; o++) {
			lol += data_in[o];
		}

		data_out[i] = lol;
	}
}

void test(unsigned int n, unsigned int deviceBatchSize, int deviceId, float* data_in, float* data_out)
{
	unsigned int numThreads = 512;

	// bigloop << <(n + numThreads - 1) / numThreads, numThreads >> > (n, deviceBatchSize, deviceId, static_cast<unsigned int>(std::min((deviceId + 1) * deviceBatchSize, n)), data_in, data_out);
}

__host__ void initializeCuda(Instance* instance)
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		gDeviceInstances.push_back(nullptr);
		cudaMalloc(&gDeviceInstances[i], instance->size());
	}
}

__host__ void unInitializeCuda()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaFree(gDeviceInstances[i]);
		cudaDeviceReset();
	}
}

__host__ void simulate(Instance* instance)
{
	Particle* particles = instance->getParticles();
	Box* boxes = instance->getBoxes();
	Particle* boxParticles = instance->getBoxParticles();

	// Build the list of particles and the center of mass for each box
	std::vector<std::vector<Particle>> tempBoxes;
	for (int i = 0; i < instance->nBoxes; i++) {
		boxes[i].mass = 0.0f;
		boxes[i].centerMass = make_float2(0.0, 0.0);

		tempBoxes.push_back(std::vector<Particle>());
	}

	// Assign each particle to a box and add its mass to the box
	for (int i = 0; i < instance->nParticles; i++) {
		int boxId = instance->getBoxIndex(particles[i].position);

		boxes[boxId].centerMass = (boxes[boxId].centerMass * boxes[boxId].mass + particles[i].position * particles[i].mass)
											/ (boxes[boxId].mass + particles[i].mass);
		boxes[boxId].mass += particles[i].mass;

		tempBoxes[boxId].push_back(particles[i]);
	}

	// Copy the particles from the temporary boxes over to the instance
	int particleOffset = 0;
	for (int i = 0; i < instance->nBoxes; i++) {
		boxes[i].nParticles = tempBoxes[i].size();
		boxes[i].particleOffset = particleOffset;
		memcpy(boxParticles + particleOffset, tempBoxes[i].data(), tempBoxes[i].size() * sizeof(Particle));
		particleOffset += tempBoxes[i].size();
	}

	int nDevices;
	cudaGetDeviceCount(&nDevices);

	// Copy the instance to device memory and run the kernel
	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		gpuErrchk(cudaMemcpy(gDeviceInstances[i], instance, instance->size(), cudaMemcpyHostToDevice));
		kernel<<<(instance->nParticles + nThreads - 1) / nThreads, nThreads>>>(i, 0, 0, gDeviceInstances[i]);
		gpuErrchk(cudaPeekAtLastError());
	}

	// Synchronize with devices and copy the udpated instance back
	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(instance, gDeviceInstances[i], instance->size(), cudaMemcpyDeviceToHost));
	}
}

__global__ void kernel(int deviceId, unsigned int deviceBatchSize, unsigned int endIndex, Instance* instance)
{
	const float time = 10.0;
	const float minForceDistance = 1.0;

	// unsigned int index = deviceId * deviceBatchSize + blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int stride = blockDim.x * gridDim.x;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	// Particle* particles = instance->getParticles();
	// Particle* boxParticles = instance->getBoxParticles();

	Particle* particles = reinterpret_cast<Particle*>(reinterpret_cast<char*>(instance) + sizeof(Instance));

	for (int i = index; i < instance->nParticles; i += stride) {
	 	float2 force = make_float2(0.0, 0.0);
	 	//for (int o = 0; o < instance->nParticles; o++) {
	 	//	// XXX/bmoody Can review making this more efficient, is it necessary to square/sqrt dist so much?
	 	//	float dist = particles[i].dist(&boxParticles[o]);
	 	//
	 	//	if (dist > minForceDistance) {
	 	//		force += (particles[i].direction(&boxParticles[o]) / dist) * ((particles[i].mass * boxParticles[o].mass) / powf(dist, 2.0));
	 	//	}
	 	//}
	 
	 	// XXX/bmoody Can consider moving this outside of the kernel
	 	// XXX/bmoody Can store force in the particle struct and avoid storing particles 2x (need to test which is faster)
	 	float2 acceleration = force / particles[i].mass;
		float test = particles[i].position.x + 10;
	 
	 	particles[i].position += (particles[i].velocity * time) + (0.5 * acceleration * powf(time, 2.0));
	 	particles[i].velocity += acceleration * time;
	 
	 	//// XXX/bmoody Review this, there must be a better way
	 	//if (particles[i].position.x < instance->left) {
	 	//	particles[i].position.x = instance->left;
	 	//	particles[i].velocity.x = 0.0;
	 	//}
	 	//if (particles[i].position.x > instance->right) {
	 	//	particles[i].position.x = instance->right - 1;
	 	//	particles[i].velocity.x = 0.0;
	 	//}
	 	//if (particles[i].position.y < instance->bottom) {
	 	//	particles[i].position.y = instance->bottom;
	 	//	particles[i].velocity.y = 0.0;
	 	//}
	 	//if (particles[i].position.y > instance->top) {
	 	//	particles[i].position.y = instance->top - 1;
	 	//	particles[i].velocity.y = 0.0;
	 	//}
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

__host__ __device__ Particle* Instance::getParticles()
{
	return reinterpret_cast<Particle*>(reinterpret_cast<char*>(this) + sizeof(Instance));
}

__host__ __device__ Box* Instance::getBoxes()
{
	return reinterpret_cast<Box*>(reinterpret_cast<char*>(this) + sizeof(Instance) + nParticles * sizeof(Particle));
}

__host__ __device__ Particle* Instance::getBoxParticles(int particleOffset)
{
	return reinterpret_cast<Particle*>(reinterpret_cast<char*>(this) + sizeof(Instance) + nParticles * sizeof(Particle) + nBoxes * sizeof(Box) + particleOffset * sizeof(Particle));
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