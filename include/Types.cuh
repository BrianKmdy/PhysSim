#pragma once

#include "cuda_runtime.h"

#include <map>
#include <string>

struct Kernel
{
	enum
	{
		unknown = 0,
		gravity,
		experimental
	};

	// static int fromString(std::string kernel);
	// static std::string toString(int kernel);

	static std::map<std::string, int> fromString;
	static std::map<int, std::string> toString;
};

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

__host__ __device__ float2 operator+(const float2& a, const float2& b);
__host__ __device__ float2 operator-(const float2& a, const float2& b);
__host__ __device__ float2 operator*(const float2& a, const float2& b);
__host__ __device__ float2 operator/(const float2& a, const float2& b);

__host__ __device__ float2 operator+(const float2& a, const float& b);
__host__ __device__ float2 operator-(const float2& a, const float& b);
__host__ __device__ float2 operator*(const float2& a, const float& b);
__host__ __device__ float2 operator/(const float2& a, const float& b);

__host__ __device__ int2 operator+(const float2& a, const int& b);
__host__ __device__ int2 operator-(const float2& a, const int& b);
__host__ __device__ int2 operator*(const float2& a, const int& b);
__host__ __device__ int2 operator/(const float2& a, const int& b);

__host__ __device__ float3 operator+(const float3& a, const float3& b);
__host__ __device__ float3 operator-(const float3& a, const float3& b);
__host__ __device__ float3 operator*(const float3& a, const float3& b);
__host__ __device__ float3 operator/(const float3& a, const float3& b);