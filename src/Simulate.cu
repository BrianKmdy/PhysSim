#include "Simulate.cuh"

const static constexpr int nThreads = 512;

std::vector<Instance> gDeviceMemory;

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

}

__host__ void unInitialize()
{

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
	for (int i = 0; i < instance->nBoxes; i++) {
		instance->boxes[i].nParticles = boxParticles[i].size();
		instance->boxes[i].particles = new Particle[boxParticles[i].size()];
		memcpy(instance->boxes[i].particles, boxParticles[i].data(), boxParticles[i].size() * sizeof(Particle));
	}

}

__global__ void kernel(Instance* instance)
{

}

__host__ __device__ unsigned int Instance::size()
{
	return 1;
}

__host__ __device__ int Instance::getBoxIndex(float2 position)
{
	int2 index = (position + (dimensions / 2)) / boxSize;

	return index.x * divisions + index.y;
}