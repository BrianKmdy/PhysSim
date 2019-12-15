#pragma once

#include "cuda_runtime.h"

#include <map>
#include <string>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__host__ __device__ float2 direction(float2 a, float2 b);
__host__ __device__ float distance(float2 a, float2 b);
__host__ __device__ float magnitude(float2 a);

__host__ __device__ int2 operator+(const int2& a, const int2& b);
__host__ __device__ int2 operator-(const int2& a, const int2& b);
__host__ __device__ int2 operator*(const int2& a, const int2& b);
__host__ __device__ int2 operator/(const int2& a, const int2& b);

__host__ __device__ int2 operator+(const int2& a, const int& b);
__host__ __device__ int2 operator-(const int2& a, const int& b);
__host__ __device__ int2 operator*(const int2& a, const int& b);
__host__ __device__ int2 operator/(const int2& a, const int& b);

__host__ __device__ int3 operator+(const int3& a, const int3& b);
__host__ __device__ int3 operator-(const int3& a, const int3& b);
__host__ __device__ int3 operator*(const int3& a, const int3& b);
__host__ __device__ int3 operator/(const int3& a, const int3& b);

__host__ __device__ float2 operator+=(float2& a, const float2& b);
__host__ __device__ float2 operator+(const float2& a, const float2& b);
__host__ __device__ float2 operator-(const float2& a, const float2& b);
__host__ __device__ float2 operator*(const float2& a, const float2& b);
__host__ __device__ float2 operator/(const float2& a, const float2& b);

__host__ __device__ float2 operator+(const float2& a, const float& b);
__host__ __device__ float2 operator-(const float2& a, const float& b);
__host__ __device__ float2 operator*(const float2& a, const float& b);
__host__ __device__ float2 operator*(const float& a, const float2& b);
__host__ __device__ float2 operator/(const float2& a, const float& b);

__host__ __device__ float2 operator+(const float2& a, const int& b);
__host__ __device__ float2 operator-(const float2& a, const int& b);
__host__ __device__ float2 operator*(const float2& a, const int& b);
__host__ __device__ float2 operator/(const float2& a, const int& b);

__host__ __device__ float3 operator+(const float3& a, const float3& b);
__host__ __device__ float3 operator-(const float3& a, const float3& b);
__host__ __device__ float3 operator*(const float3& a, const float3& b);
__host__ __device__ float3 operator/(const float3& a, const float3& b);

__host__ __device__ int2 abs(const int2& a);