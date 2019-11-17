#include "Simulate.cuh"

const static constexpr int nThreads = 512;

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

__host__ void initialize(Instance* instance)
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		gDeviceInstances.push_back(nullptr);
		cudaMalloc(&gDeviceInstances[i], instance->size());
	}
}

__host__ void unInitialize()
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

	// Build the list of particles and the center of mass for each box
	std::vector<std::vector<Particle>> boxParticles;
	for (int i = 0; i < instance->nBoxes; i++) {
		boxes[i].mass = 0.0f;
		boxes[i].centerMass = make_float2(0.0, 0.0);

		boxParticles.push_back(std::vector<Particle>());
	}

	// Assign each particle to a box and add its mass to the box
	for (int i = 0; i < instance->nParticles; i++) {
		int boxId = instance->getBoxIndex(particles[i].position);

		boxes[boxId].centerMass = (boxes[boxId].centerMass * boxes[boxId].mass + particles[i].position * particles[i].mass)
											/ (boxes[boxId].mass + particles[i].mass);
		boxes[boxId].mass += particles[i].mass;

		boxParticles[boxId].push_back(particles[i]);
	}

	// Copy the particles from the temporary boxes over to the instance
	int particleOffset = 0;
	for (int i = 0; i < instance->nBoxes; i++) {
		boxes[i].nParticles = boxParticles[i].size();
		boxes[i].particleOffset = particleOffset;
		particleOffset += boxParticles[i].size();
	}

	int nDevices;
	cudaGetDeviceCount(&nDevices);

	// Copy the instance to device memory and run the kernel
	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		cudaMemcpy(gDeviceInstances[i], instance, instance->size(), cudaMemcpyHostToDevice);
		kernel<<<(instance->nParticles + nThreads - 1) / nThreads, nThreads>>> (i, 0, 0, gDeviceInstances[i]);
	}

	// Synchronize with devices and copy the udpated instance back
	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		cudaDeviceSynchronize();
		cudaMemcpy(instance, gDeviceInstances[i], instance->size(), cudaMemcpyDeviceToHost);
	}
}

__global__ void kernel(int deviceId, unsigned int deviceBatchSize, unsigned int endIndex, Instance* instance)
{
	Particle* particles = instance->getParticles();
	// unsigned int index = deviceId * deviceBatchSize + blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int stride = blockDim.x * gridDim.x;

	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = index; i < instance->nParticles; i += stride) {
		particles[i].mass = i;
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