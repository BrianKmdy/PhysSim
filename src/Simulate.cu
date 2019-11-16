#include "Simulate.cuh"

const static constexpr int nThreads = 512;

std::vector<Instance*> gDeviceMemory;

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

		gDeviceMemory.push_back(nullptr);
		cudaMalloc(&gDeviceMemory[i], instance->size());
	}
}

__host__ void unInitialize()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaFree(gDeviceMemory[i]);
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

	for (int i = 0; i < instance->nParticles; i++) {
		int boxId = instance->getBoxIndex(instance->particles[i].position);

		instance->boxes[boxId].centerMass = (instance->boxes[boxId].centerMass * instance->boxes[boxId].mass + instance->particles[i].position * instance->particles[i].mass)
											/ (instance->boxes[boxId].mass + instance->particles[i].mass);
		instance->boxes[boxId].mass += instance->particles[i].mass;

		boxParticles[boxId].push_back(instance->particles[i]);
	}

	// XXX/bmoody For testing
	Particle* particlePointer = instance->boxParticles;
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

}

__global__ void kernel(Instance* instance)
{

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

__host__ void Instance::initialize()
{
	char* pointer = reinterpret_cast<char*>(this);
	pointer += sizeof(Instance);

	particles = reinterpret_cast<Particle*>(pointer);
	pointer += nParticles * sizeof(Particle);

	boxes = reinterpret_cast<Box*>(pointer);
	pointer += nBoxes * sizeof(Box);

	boxParticles = reinterpret_cast<Particle*>(pointer);
}

__host__ void Instance::copyFull(Instance* instance)
{
	memcpy(instance, this, size());

	// Initialize the particles and boxes pointers
	instance->initialize();

	// Copy over the list of particles for each box
	Particle* particlePointer = instance->boxParticles;
	for (int i = 0; i < nBoxes; i++) {
		if (boxes[i].nParticles > 0) {
			instance->boxes[i].particles = particlePointer;
			particlePointer += boxes[i].nParticles;
		}
		else {
			instance->boxes[i].particles = nullptr;
		}
	}
}
