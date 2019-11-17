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
	// Build the list of particles and the center of mass for each box
	std::vector<std::vector<Particle>> boxParticles;
	for (int i = 0; i < instance->nBoxes; i++) {
		instance->boxes[i].mass = 0.0f;
		instance->boxes[i].centerMass = make_float2(0.0, 0.0);

		boxParticles.push_back(std::vector<Particle>());
	}

	// Assign each particle to a box and add its mass to the box
	for (int i = 0; i < instance->nParticles; i++) {
		int boxId = instance->getBoxIndex(instance->particles[i].position);

		instance->boxes[boxId].centerMass = (instance->boxes[boxId].centerMass * instance->boxes[boxId].mass + instance->particles[i].position * instance->particles[i].mass)
											/ (instance->boxes[boxId].mass + instance->particles[i].mass);
		instance->boxes[boxId].mass += instance->particles[i].mass;

		boxParticles[boxId].push_back(instance->particles[i]);
	}

	// Copy the particles from the temporary boxes over to the instance
	Particle* particlePointer = reinterpret_cast<Particle*>(instance->boxes + instance->nBoxes);
	for (int i = 0; i < instance->nBoxes; i++) {
		instance->boxes[i].nParticles = boxParticles[i].size();

		if (boxParticles[i].size() > 0) {
			instance->boxes[i].particles = particlePointer;
			memcpy(particlePointer, boxParticles[i].data(), boxParticles[i].size() * sizeof(Particle));
			particlePointer += boxParticles[i].size();
		}
		else {
			instance->boxes[i].particles = nullptr;
		}
	}

	int nDevices;
	cudaGetDeviceCount(&nDevices);

	// Copy the instance to device memory and run the kernel
	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		cudaMemcpy(gDeviceInstances[i], instance, instance->size(), cudaMemcpyHostToDevice);
		initializeInstance(gDeviceInstances[i]);

		kernel<<<(instance->nParticles + nThreads - 1) / nThreads, nThreads>>> (i, 0, 0, gDeviceInstances[i]);
	}

	// Synchronize with devices and copy the udpated instance back
	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);

		cudaDeviceSynchronize();

		cudaMemcpy(instance, gDeviceInstances[i], instance->size(), cudaMemcpyDeviceToHost);
		initializeInstance(instance);
	}
}

__global__ void kernel(int deviceId, unsigned int deviceBatchSize, unsigned int endIndex, Instance* instance)
{
	// unsigned int index = deviceId * deviceBatchSize + blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int stride = blockDim.x * gridDim.x;

	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = index; i < instance->nParticles; i += stride) {
		instance->particles[i].mass = i;
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

__host__ void initializeInstance(Instance* instance)
{
	char* pointer = reinterpret_cast<char*>(instance);
	pointer += sizeof(Instance);

	instance->particles = reinterpret_cast<Particle*>(pointer);
	pointer += instance->nParticles * sizeof(Particle);

	instance->boxes = reinterpret_cast<Box*>(pointer);
	pointer += instance->nBoxes * sizeof(Box);

	for (int i = 0; i < instance->nBoxes; i++) {
		if (instance->boxes[i].nParticles > 0) {
			instance->boxes[i].particles = reinterpret_cast<Particle*>(pointer);
			pointer += instance->boxes[i].nParticles * sizeof(Particle);
		}
		else {
			instance->boxes[i].particles = nullptr;
		}
	}
}