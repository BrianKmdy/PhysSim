#include <algorithm>

#include "kernel.cuh"

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

	bigloop << <(n + numThreads - 1) / numThreads, numThreads >> > (n, deviceBatchSize, deviceId, static_cast<unsigned int>(std::min((deviceId + 1) * deviceBatchSize, n)), data_in, data_out);
}

__host__ __device__ int test_math()
{
	return 5;
}
